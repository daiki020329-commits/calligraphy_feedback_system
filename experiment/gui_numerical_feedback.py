"""
条件D: 数値可視化フィードバック GUI

LLMを使わず、グラフと数値のみで構成されるフィードバックを提示する。

フィードバック内容（筆圧・速度それぞれヒートマップとグラフを横並び表示）:
- 筆圧ヒートマップ（お手本 vs ユーザー）+ 筆圧ストローク別比較グラフ
- 速度ヒートマップ（お手本 vs ユーザー）+ 速度ストローク別比較グラフ
- 各筆画の DTW 座標一致率テーブル

保存先: experiment/user_data_numerical_feedback/

使い方:
    python experiment/gui_numerical_feedback.py
"""

import glob
import io
import json
import math
import os
import sys
import time
import traceback

import numpy as np

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QBuffer, QIODevice, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QImage

from calligraphy_input_app import CalligraphyCanvas
from experiment.scoring import _extract_xy_sequences

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# スタイル定数
# ---------------------------------------------------------------------------

BUTTON_STYLE_ORANGE = """
    QPushButton {
        background-color: #ff9800; color: white; border: none;
        padding: 8px 18px; border-radius: 5px;
        font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background-color: #f57c00; }
    QPushButton:disabled { background-color: #ccc; color: #888; }
"""

BUTTON_STYLE_RED = """
    QPushButton {
        background-color: #f44336; color: white; border: none;
        padding: 8px 18px; border-radius: 5px;
        font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background-color: #d32f2f; }
    QPushButton:disabled { background-color: #ccc; color: #888; }
"""

BUTTON_STYLE_GREEN = """
    QPushButton {
        background-color: #4CAF50; color: white; border: none;
        padding: 8px 18px; border-radius: 5px;
        font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background-color: #388E3C; }
    QPushButton:disabled { background-color: #ccc; color: #888; }
"""


# ---------------------------------------------------------------------------
# QPainter ベースのレンダリング（gui_app.py と同一ロジック）
# ---------------------------------------------------------------------------

_MIN_WIDTH = 1.5
_MAX_WIDTH = 30.0
_PRESSURE_EXP = 2.0
_INK_COLOR = (20, 20, 20)
_PAPER_COLOR = QColor(252, 250, 245)
_BORDER_COLOR = QColor(180, 170, 160)


def _smooth_pressure(stroke_data: list, window_size: int = 3) -> list[float]:
    if len(stroke_data) <= window_size:
        return [pt[2] for pt in stroke_data]
    pressures = [pt[2] for pt in stroke_data]
    smoothed = []
    for i in range(len(pressures)):
        start = max(0, i - window_size // 2)
        end = min(len(pressures), i + window_size // 2 + 1)
        smoothed.append(sum(pressures[start:end]) / (end - start))
    return smoothed


def _draw_brush_segment_qp(
    painter: QPainter,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width1: float,
    width2: float,
    alpha: int,
):
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 0.5:
        return
    segments = max(3, int(length / 3))
    for i in range(segments):
        t = i / segments
        t_next = (i + 1) / segments
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        nx = x1 + t_next * (x2 - x1)
        ny = y1 + t_next * (y2 - y1)
        cw = width1 + t * (width2 - width1)
        nw = width1 + t_next * (width2 - width1)
        avg_w = (cw + nw) / 2

        pen = QPen(QColor(*_INK_COLOR, alpha))
        pen.setWidthF(max(1.0, avg_w))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(int(cx), int(cy), int(nx), int(ny))

        if avg_w > 8:
            soft = QPen(QColor(30, 30, 30, alpha // 3))
            soft.setWidthF(avg_w * 1.15)
            soft.setCapStyle(Qt.RoundCap)
            painter.setPen(soft)
            painter.drawLine(int(cx), int(cy), int(nx), int(ny))


def _draw_stroke_qp(painter: QPainter, stroke: list, canvas_size: int):
    if len(stroke) < 2:
        return
    smoothed = _smooth_pressure(stroke)
    for i in range(len(stroke) - 1):
        nx1, ny1, _, _ = stroke[i]
        nx2, ny2, _, _ = stroke[i + 1]
        p1, p2 = smoothed[i], smoothed[i + 1]
        x1 = nx1 * canvas_size
        y1 = ny1 * canvas_size
        x2 = nx2 * canvas_size
        y2 = ny2 * canvas_size
        w1 = _MIN_WIDTH + (p1**_PRESSURE_EXP) * (_MAX_WIDTH - _MIN_WIDTH)
        w2 = _MIN_WIDTH + (p2**_PRESSURE_EXP) * (_MAX_WIDTH - _MIN_WIDTH)
        alpha = 150 + int((p1**1.5) * 105)
        _draw_brush_segment_qp(painter, x1, y1, x2, y2, w1, w2, alpha)


def _pixmap_to_png(pixmap: QPixmap) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    return bytes(buf.data())


def _render_strokes_to_png(stroke_data: dict, size: int = 600) -> bytes:
    pixmap = QPixmap(size, size)
    pixmap.fill(_PAPER_COLOR)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(_BORDER_COLOR, 2))
    painter.drawRect(0, 0, size - 1, size - 1)
    for stroke in stroke_data.get("strokes", []):
        _draw_stroke_qp(painter, stroke, size)
    painter.end()
    return _pixmap_to_png(pixmap)


# ---------------------------------------------------------------------------
# 数値フィードバック生成ヘルパー
# ---------------------------------------------------------------------------


def _compute_speed_per_stroke(strokes: list) -> list[np.ndarray]:
    """各ストロークのポイントごとの速度を計算する。

    Returns:
        ストロークごとの速度配列のリスト。各配列は 0〜1 に正規化された
        進捗率に対応する速度値を持つ。
    """
    stroke_speeds = []

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        arr = np.array(stroke, dtype=float)
        xs, ys, _, ts = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        raw_speeds = [0.0]
        for i in range(1, len(arr)):
            dt = ts[i] - ts[i - 1]
            if dt <= 0:
                raw_speeds.append(raw_speeds[-1] if raw_speeds else 0.0)
                continue
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            dist = np.sqrt(dx**2 + dy**2)
            speed = dist / (dt / 1000.0)  # 正規化座標/秒
            raw_speeds.append(speed)

        speeds = np.array(raw_speeds)

        # 移動平均スムージング（window=5）でスパイクを除去
        if len(speeds) >= 5:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(speeds, kernel, mode="same")
            # 端の処理: 最初と最後の2点は小さいカーネルで
            for i in range(2):
                w = 2 * i + 1
                smoothed[i] = np.mean(speeds[: w + 1])
                smoothed[-(i + 1)] = np.mean(speeds[-(w + 1) :])
            speeds = smoothed

        stroke_speeds.append(speeds)

    return stroke_speeds


def _draw_heatmap_on_ax(
    ax, strokes: list, value_index: int, cmap_name: str, label: str
):
    """ストロークの軌跡上に指定値をカラーマップで描画する共通関数。

    Args:
        ax: matplotlib の Axes
        strokes: ストロークデータ
        value_index: 各ポイントから取る値のインデックス (2=筆圧)
        cmap_name: カラーマップ名
        label: カラーバーのラベル
    """
    cmap = matplotlib.colormaps[cmap_name]
    # 全ストロークから値の範囲を取得
    all_vals = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        arr = np.array(stroke, dtype=float)
        all_vals.extend(arr[:, value_index].tolist())
    if not all_vals:
        return
    vmin, vmax = min(all_vals), max(all_vals)
    if vmax - vmin < 1e-9:
        vmin, vmax = 0, max(vmax, 1e-9)
    norm = plt.Normalize(vmin, vmax)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        arr = np.array(stroke, dtype=float)
        xs, ys, vals = arr[:, 0], arr[:, 1], arr[:, value_index]
        for i in range(len(arr) - 1):
            ax.plot(
                [xs[i], xs[i + 1]],
                [ys[i], ys[i + 1]],
                color=cmap(norm(vals[i])),
                linewidth=max(1.0, norm(vals[i]) * 5),
                solid_capstyle="round",
            )

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def _generate_pressure_heatmap_png(ref_data: dict, user_data: dict) -> bytes:
    """お手本とユーザーの筆圧ヒートマップを並べて生成する。"""
    fig = Figure(figsize=(8, 4), dpi=150)

    for col, (data, title) in enumerate(
        [
            (ref_data, "お手本 - 筆圧"),
            (user_data, "ユーザー - 筆圧"),
        ]
    ):
        ax = fig.add_subplot(1, 2, col + 1)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_facecolor("#fcfaf5")
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=9)
        _draw_heatmap_on_ax(ax, data.get("strokes", []), 2, "coolwarm", "筆圧")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_pressure_comparison_png(ref_data: dict, user_data: dict) -> bytes:
    """お手本 vs ユーザーの筆圧比較折れ線グラフをストロークごとに生成する。"""
    ref_strokes = ref_data.get("strokes", [])
    user_strokes = user_data.get("strokes", [])

    def _smooth_array(arr: np.ndarray, window: int = 5) -> np.ndarray:
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        smoothed = np.convolve(arr, kernel, mode="same")
        half = window // 2
        for i in range(half):
            w = 2 * i + 1
            smoothed[i] = np.mean(arr[: w + 1])
            smoothed[-(i + 1)] = np.mean(arr[-(w + 1) :])
        return smoothed

    ref_pressures = []
    for stroke in ref_strokes:
        if len(stroke) < 2:
            continue
        ref_pressures.append(_smooth_array(np.array([pt[2] for pt in stroke])))

    user_pressures = []
    for stroke in user_strokes:
        if len(stroke) < 2:
            continue
        user_pressures.append(_smooth_array(np.array([pt[2] for pt in stroke])))

    n_strokes = max(len(ref_pressures), len(user_pressures))
    if n_strokes == 0:
        n_strokes = 1

    n_cols = min(n_strokes, 3)
    n_rows = math.ceil(n_strokes / n_cols)
    fig = Figure(figsize=(4.5 * n_cols, 3.0 * n_rows + 0.6), dpi=150)
    fig.suptitle("筆圧比較（ストローク別）", fontsize=14, y=0.98)

    for si in range(n_strokes):
        ax = fig.add_subplot(n_rows, n_cols, si + 1)

        if si < len(ref_pressures):
            rp = ref_pressures[si]
            progress = np.linspace(0, 100, len(rp))
            ax.plot(
                progress, rp, color="#2196F3", alpha=0.8, linewidth=2.0, label="お手本"
            )
            ax.fill_between(progress, rp, alpha=0.1, color="#2196F3")

        if si < len(user_pressures):
            up = user_pressures[si]
            progress = np.linspace(0, 100, len(up))
            ax.plot(
                progress,
                up,
                color="#F44336",
                alpha=0.8,
                linewidth=2.0,
                label="ユーザー",
            )
            ax.fill_between(progress, up, alpha=0.1, color="#F44336")

        ax.set_title(f"第{si + 1}画", fontsize=13)
        ax.set_xlabel("進捗 (%)", fontsize=11)
        if si % n_cols == 0:
            ax.set_ylabel("筆圧", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        if si == 0:
            ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_speed_heatmap_png(ref_data: dict, user_data: dict) -> bytes:
    """お手本とユーザーの速度ヒートマップを並べて生成する。

    軌跡上の各セグメントの速度を色で表現する。
    """

    # まずストロークごとの速度を計算し、各ポイントに速度値を付与
    def _attach_speed_to_strokes(strokes):
        """ストロークデータに速度を付与した拡張ストロークを返す。
        各ポイント: [x, y, speed, time] の形式。"""
        result = []
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            arr = np.array(stroke, dtype=float)
            xs, ys, _, ts = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            speeds = [0.0]
            for i in range(1, len(arr)):
                dt = ts[i] - ts[i - 1]
                if dt <= 0:
                    speeds.append(speeds[-1] if speeds else 0.0)
                    continue
                dist = np.sqrt((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2)
                speeds.append(dist / (dt / 1000.0))
            # スムージング
            speeds = np.array(speeds)
            if len(speeds) >= 5:
                kernel = np.ones(5) / 5
                smoothed = np.convolve(speeds, kernel, mode="same")
                for i in range(2):
                    w = 2 * i + 1
                    smoothed[i] = np.mean(speeds[: w + 1])
                    smoothed[-(i + 1)] = np.mean(speeds[-(w + 1) :])
                speeds = smoothed
            # [x, y, speed, time] として再構成
            extended = [[xs[i], ys[i], speeds[i], ts[i]] for i in range(len(arr))]
            result.append(extended)
        return result

    ref_speed_strokes = _attach_speed_to_strokes(ref_data.get("strokes", []))
    user_speed_strokes = _attach_speed_to_strokes(user_data.get("strokes", []))

    # 全データの速度範囲をパーセンタイルで決定（外れ値に引っ張られない）
    all_speeds = []
    for s in ref_speed_strokes + user_speed_strokes:
        all_speeds.extend([pt[2] for pt in s])
    if not all_speeds:
        all_speeds = [0.0, 1.0]
    all_speeds_arr = np.array(all_speeds)
    # 5〜95パーセンタイルで範囲を決定
    vmin = float(np.percentile(all_speeds_arr, 5))
    vmax = float(np.percentile(all_speeds_arr, 95))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    # ガンマ補正（gamma<1で低速側の差を強調）
    from matplotlib.colors import PowerNorm

    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True)

    fig = Figure(figsize=(8, 4), dpi=150)
    cmap = matplotlib.colormaps["plasma"]

    for col, (speed_strokes, title) in enumerate(
        [
            (ref_speed_strokes, "お手本 - 速度"),
            (user_speed_strokes, "ユーザー - 速度"),
        ]
    ):
        ax = fig.add_subplot(1, 2, col + 1)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_facecolor("#fcfaf5")
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=9)

        for stroke in speed_strokes:
            arr = np.array(stroke, dtype=float)
            xs, ys, spds = arr[:, 0], arr[:, 1], arr[:, 2]
            for i in range(len(arr) - 1):
                nv = float(norm(spds[i]))
                ax.plot(
                    [xs[i], xs[i + 1]],
                    [ys[i], ys[i + 1]],
                    color=cmap(nv),
                    linewidth=1.5 + nv * 3.5,
                    solid_capstyle="round",
                )

        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("速度", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_speed_comparison_png(
    ref_data: dict,
    user_data: dict,
) -> bytes:
    """お手本 vs ユーザーの速度比較折れ線グラフをストロークごとに生成する。

    各ストロークを 0〜100% の進捗率に正規化し、同じ画同士を
    直接比較できるサブプロットとして並べる。
    """
    ref_stroke_speeds = _compute_speed_per_stroke(ref_data.get("strokes", []))
    user_stroke_speeds = _compute_speed_per_stroke(user_data.get("strokes", []))

    n_strokes = max(len(ref_stroke_speeds), len(user_stroke_speeds))
    if n_strokes == 0:
        n_strokes = 1

    # レイアウト: 最大3列
    n_cols = min(n_strokes, 3)
    n_rows = math.ceil(n_strokes / n_cols)
    fig_w = 4.5 * n_cols
    fig_h = 3.0 * n_rows + 0.6  # タイトル分の余白

    fig = Figure(figsize=(fig_w, fig_h), dpi=150)
    fig.suptitle("速度比較（ストローク別）", fontsize=14, y=0.98)

    for si in range(n_strokes):
        ax = fig.add_subplot(n_rows, n_cols, si + 1)

        # お手本
        if si < len(ref_stroke_speeds):
            ref_s = ref_stroke_speeds[si]
            ref_progress = np.linspace(0, 100, len(ref_s))
            ax.plot(
                ref_progress,
                ref_s,
                color="#2196F3",
                alpha=0.8,
                linewidth=2.0,
                label="お手本",
            )
            ax.fill_between(ref_progress, ref_s, alpha=0.1, color="#2196F3")

        # ユーザー
        if si < len(user_stroke_speeds):
            user_s = user_stroke_speeds[si]
            user_progress = np.linspace(0, 100, len(user_s))
            ax.plot(
                user_progress,
                user_s,
                color="#F44336",
                alpha=0.8,
                linewidth=2.0,
                label="ユーザー",
            )
            ax.fill_between(user_progress, user_s, alpha=0.1, color="#F44336")

        ax.set_title(f"第{si + 1}画", fontsize=13)
        ax.set_xlabel("進捗 (%)", fontsize=11)
        if si % n_cols == 0:
            ax.set_ylabel("速度", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        if si == 0:
            ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _compute_per_stroke_dtw_similarity(ref_data: dict, user_data: dict) -> list[dict]:
    """各ストロークの DTW 座標類似度をパーセンテージで返す。

    Returns:
        [{"stroke": 1, "similarity_pct": 85.2}, ...]
    """
    from dtw import dtw

    ref_seqs = _extract_xy_sequences(ref_data)
    user_seqs = _extract_xy_sequences(user_data)

    results = []
    n_pairs = min(len(ref_seqs), len(user_seqs))

    for i in range(n_pairs):
        alignment = dtw(ref_seqs[i], user_seqs[i])
        similarity = 1.0 / (1.0 + alignment.distance)
        results.append(
            {
                "stroke": i + 1,
                "similarity_pct": round(similarity * 100, 1),
            }
        )

    # ユーザーの余分なストローク
    for i in range(n_pairs, len(user_seqs)):
        results.append(
            {
                "stroke": i + 1,
                "similarity_pct": 0.0,
            }
        )

    return results


# ---------------------------------------------------------------------------
# NumericalFeedbackEntry: 1回分のフィードバック表示ウィジェット
# ---------------------------------------------------------------------------


class NumericalFeedbackEntry(QFrame):
    def __init__(
        self,
        attempt_number: int,
        pressure_heatmap_png: bytes,
        pressure_graph_png: bytes,
        speed_heatmap_png: bytes,
        speed_graph_png: bytes,
        dtw_results: list[dict],
        parent=None,
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "NumericalFeedbackEntry { background-color: #ffffff; border: 1px solid #ddd; "
            "border-radius: 6px; margin: 4px 0; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        # ヘッダー
        header = QLabel(f"Attempt #{attempt_number}")
        header.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #333; border: none;"
        )
        layout.addWidget(header)

        # 画像をQLabelに変換するヘルパー（アスペクト比を維持）
        def _make_image_label(png_data: bytes, max_width: int = 0) -> QLabel:
            label = QLabel()
            label.setStyleSheet("border: none;")
            label.setAlignment(Qt.AlignCenter)
            qimg = QImage.fromData(png_data)
            if not qimg.isNull():
                pixmap = QPixmap.fromImage(qimg)
                if max_width > 0 and pixmap.width() > max_width:
                    pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
                label.setPixmap(pixmap)
            return label

        # --- 筆圧セクション: ヒートマップ（左） + グラフ（右） ---
        pressure_title = QLabel("筆圧")
        pressure_title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #1565C0; border: none; padding-top: 2px;"
        )
        layout.addWidget(pressure_title)

        pressure_row = QHBoxLayout()
        pressure_row.setSpacing(8)
        pressure_row.setAlignment(Qt.AlignTop)

        heatmap_lbl = _make_image_label(pressure_heatmap_png)
        pressure_row.addWidget(heatmap_lbl, stretch=2)

        graph_lbl = _make_image_label(pressure_graph_png)
        pressure_row.addWidget(graph_lbl, stretch=3)

        layout.addLayout(pressure_row)

        # --- 速度セクション: ヒートマップ（左） + グラフ（右） ---
        speed_title = QLabel("速度")
        speed_title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #E65100; border: none; padding-top: 6px;"
        )
        layout.addWidget(speed_title)

        speed_row = QHBoxLayout()
        speed_row.setSpacing(8)
        speed_row.setAlignment(Qt.AlignTop)

        speed_heatmap_lbl = _make_image_label(speed_heatmap_png)
        speed_row.addWidget(speed_heatmap_lbl, stretch=2)

        speed_graph_lbl = _make_image_label(speed_graph_png)
        speed_row.addWidget(speed_graph_lbl, stretch=3)

        layout.addLayout(speed_row)

        # --- 各筆画の一致率テーブル ---
        if dtw_results:
            table_header = QLabel("各筆画の一致率")
            table_header.setStyleSheet(
                "font-weight: bold; font-size: 14px; color: #555; "
                "padding-top: 6px; border: none;"
            )
            layout.addWidget(table_header)

            lines = []
            for r in dtw_results:
                pct = r["similarity_pct"]
                bar_len = int(pct / 5)  # 最大20文字
                bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
                lines.append(f"第{r['stroke']}画: {pct:5.1f}%  {bar}")

            table_label = QLabel("\n".join(lines))
            table_label.setStyleSheet(
                "font-size: 14px; color: #222; font-family: 'Consolas', 'Courier New', monospace; "
                "padding: 6px 10px; border: none; background-color: #f9f9f9; border-radius: 4px;"
            )
            table_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(table_label)


# ---------------------------------------------------------------------------
# NumericalFeedbackWorker: バックグラウンド計算スレッド
# ---------------------------------------------------------------------------


class NumericalFeedbackWorker(QThread):
    # pressure_heatmap, pressure_graph, speed_heatmap, speed_graph, dtw_results
    finished = pyqtSignal(bytes, bytes, bytes, bytes, list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, ref_data: dict, user_data: dict):
        super().__init__()
        self.ref_data = ref_data
        self.user_data = user_data

    def run(self):
        try:
            self.progress.emit("筆圧ヒートマップを生成中...")
            pressure_heatmap = _generate_pressure_heatmap_png(
                self.ref_data, self.user_data
            )

            self.progress.emit("筆圧比較グラフを生成中...")
            pressure_graph = _generate_pressure_comparison_png(
                self.ref_data, self.user_data
            )

            self.progress.emit("速度ヒートマップを生成中...")
            speed_heatmap = _generate_speed_heatmap_png(self.ref_data, self.user_data)

            self.progress.emit("速度比較グラフを生成中...")
            speed_graph = _generate_speed_comparison_png(self.ref_data, self.user_data)

            self.progress.emit("DTW一致率を計算中...")
            dtw_results = _compute_per_stroke_dtw_similarity(
                self.ref_data, self.user_data
            )

            self.finished.emit(
                pressure_heatmap,
                pressure_graph,
                speed_heatmap,
                speed_graph,
                dtw_results,
            )

        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# NumericalFeedbackApp: メインウィンドウ
# ---------------------------------------------------------------------------


class NumericalFeedbackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("書道フィードバック（条件: 数値可視化）")
        self.setGeometry(20, 20, 1920, 1000)

        self.reference_data: dict | None = None
        self.attempt_number: int = 0
        self.worker: NumericalFeedbackWorker | None = None

        self._ref_map: dict[str, str] = {}

        self._build_ui()
        self._scan_references()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 0)
        root_layout.setSpacing(6)

        # 参照文字選択
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("参照文字:"))
        self.char_combo = QComboBox()
        self.char_combo.setMinimumWidth(200)
        self.char_combo.currentIndexChanged.connect(self._on_reference_changed)
        top_bar.addWidget(self.char_combo)
        top_bar.addStretch()
        root_layout.addLayout(top_bar)

        # メイン: 左右スプリッター
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # 左パネル
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.ref_image_label = QLabel("お手本画像")
        self.ref_image_label.setFixedSize(300, 300)
        self.ref_image_label.setAlignment(Qt.AlignCenter)
        self.ref_image_label.setStyleSheet(
            "background-color: #fcfaf5; border: 1px solid #bbb; border-radius: 4px;"
        )
        left_layout.addWidget(self.ref_image_label, 0, Qt.AlignHCenter)

        self.canvas = CalligraphyCanvas()
        self.canvas.setMinimumSize(600, 600)
        self.canvas.setMaximumSize(600, 600)
        left_layout.addWidget(self.canvas, 0, Qt.AlignHCenter)

        btn_row = QHBoxLayout()
        self.undo_btn = QPushButton("1画戻る")
        self.undo_btn.setStyleSheet(BUTTON_STYLE_ORANGE)
        self.undo_btn.clicked.connect(self._on_undo)
        btn_row.addWidget(self.undo_btn)

        self.clear_btn = QPushButton("全消去")
        self.clear_btn.setStyleSheet(BUTTON_STYLE_RED)
        self.clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()

        self.submit_btn = QPushButton("送信")
        self.submit_btn.setStyleSheet(BUTTON_STYLE_GREEN)
        self.submit_btn.clicked.connect(self._on_submit)
        btn_row.addWidget(self.submit_btn)

        left_layout.addLayout(btn_row)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # 右パネル（フィードバック履歴）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        right_header = QLabel("フィードバック履歴")
        right_header.setStyleSheet(
            "font-weight: bold; font-size: 15px; padding: 6px 4px; color: #333;"
        )
        right_layout.addWidget(right_header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: #f5f5f5; }"
        )

        self.feedback_container = QWidget()
        self.feedback_layout = QVBoxLayout(self.feedback_container)
        self.feedback_layout.setContentsMargins(6, 6, 6, 6)
        self.feedback_layout.setSpacing(8)
        self.feedback_layout.addStretch()

        self.scroll_area.setWidget(self.feedback_container)
        right_layout.addWidget(self.scroll_area)

        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([640, 1220])

        root_layout.addWidget(splitter, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了")

    # ---------------------------------------------------------------
    # 参照データ管理
    # ---------------------------------------------------------------

    def _scan_references(self):
        ref_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reference_data",
        )
        if not os.path.isdir(ref_dir):
            self.status_bar.showMessage("reference_data ディレクトリが見つかりません")
            return

        json_files = sorted(glob.glob(os.path.join(ref_dir, "*.json")))
        if not json_files:
            self.status_bar.showMessage("参照データが見つかりません")
            return

        self.char_combo.blockSignals(True)
        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                char = data.get("character", os.path.basename(path))
                display = f"{char}  ({os.path.basename(path)})"
                self._ref_map[display] = path
                self.char_combo.addItem(display)
            except Exception:
                continue
        self.char_combo.blockSignals(False)

        if self.char_combo.count() > 0:
            self._on_reference_changed(0)

    def _on_reference_changed(self, index: int):
        display = self.char_combo.currentText()
        path = self._ref_map.get(display)
        if path is None:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.reference_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"参照データの読み込みに失敗:\n{e}")
            return

        self.attempt_number = 0
        self._clear_feedback_entries()
        self.canvas.clear_canvas()
        self._update_ref_image()
        self.status_bar.showMessage(
            f"参照文字「{self.reference_data.get('character', '?')}」を読み込みました"
        )

    def _update_ref_image(self):
        if self.reference_data is None:
            return
        try:
            png = _render_strokes_to_png(self.reference_data, size=600)
            qimg = QImage.fromData(png)
            if not qimg.isNull():
                pixmap = QPixmap.fromImage(qimg)
                pixmap = pixmap.scaled(
                    300,
                    300,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.ref_image_label.setPixmap(pixmap)
        except Exception:
            self.ref_image_label.setText("(画像生成エラー)")

    # ---------------------------------------------------------------
    # キャンバス操作
    # ---------------------------------------------------------------

    def _on_undo(self):
        if not self.canvas.undo_last_stroke():
            self.status_bar.showMessage("削除するストロークがありません")

    def _on_clear(self):
        if self.canvas.get_stroke_count() == 0:
            return
        self.canvas.clear_canvas()
        self.status_bar.showMessage("キャンバスをクリアしました")

    # ---------------------------------------------------------------
    # 送信 → 数値フィードバック
    # ---------------------------------------------------------------

    def _on_submit(self):
        if self.reference_data is None:
            QMessageBox.warning(self, "警告", "参照データが選択されていません。")
            return

        user_data = self.canvas.get_all_data()
        if user_data["stroke_count"] == 0:
            QMessageBox.warning(
                self, "警告", "キャンバスに何か書いてから送信してください。"
            )
            return

        self.submit_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        self.attempt_number += 1
        self._current_user_data = user_data

        self.status_bar.showMessage("数値フィードバックを生成中...")

        self.worker = NumericalFeedbackWorker(self.reference_data, user_data)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, message: str):
        self.status_bar.showMessage(message)

    def _on_worker_finished(
        self,
        pressure_heatmap: bytes,
        pressure_graph: bytes,
        speed_heatmap: bytes,
        speed_graph: bytes,
        dtw_results: list,
    ):
        saved_path = self._save_attempt(self._current_user_data, dtw_results)

        entry = NumericalFeedbackEntry(
            self.attempt_number,
            pressure_heatmap,
            pressure_graph,
            speed_heatmap,
            speed_graph,
            dtw_results,
        )
        idx = self.feedback_layout.count() - 1
        self.feedback_layout.insertWidget(idx, entry)

        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(
            100,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

        self.canvas.clear_canvas()
        self.submit_btn.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        status = f"Attempt #{self.attempt_number} のフィードバックを表示しました"
        if saved_path:
            status += f" (保存: {os.path.basename(saved_path)})"
        self.status_bar.showMessage(status)

        self.worker = None

    def _on_worker_error(self, error_text: str):
        self.attempt_number -= 1
        self.submit_btn.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.status_bar.showMessage("エラーが発生しました")
        QMessageBox.critical(
            self, "エラー", f"フィードバック生成に失敗しました:\n\n{error_text}"
        )
        self.worker = None

    # ---------------------------------------------------------------
    # ユーティリティ
    # ---------------------------------------------------------------

    def _save_attempt(self, user_data: dict, dtw_results: list) -> str | None:
        try:
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "user_data_numerical_feedback",
            )
            os.makedirs(save_dir, exist_ok=True)

            character = self.reference_data.get("character", "unknown")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{character}_{timestamp}_{self.attempt_number:03d}.json"
            path = os.path.join(save_dir, filename)

            record = {
                "character": character,
                "attempt_number": self.attempt_number,
                "timestamp": timestamp,
                "strokes": user_data["strokes"],
                "stroke_count": user_data["stroke_count"],
                "canvas_size": user_data["canvas_size"],
                "dtw_per_stroke": dtw_results,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            return path
        except Exception as e:
            self.status_bar.showMessage(f"保存失敗: {e}")
            return None

    def _clear_feedback_entries(self):
        while self.feedback_layout.count() > 1:
            item = self.feedback_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("書道フィードバック（数値可視化）")

    font = QFont()
    font.setFamily("Yu Gothic UI" if sys.platform == "win32" else "Hiragino Sans")
    font.setPointSize(10)
    app.setFont(font)

    # matplotlib の日本語フォント設定
    plt.rcParams["font.family"] = (
        "Yu Gothic" if sys.platform == "win32" else "Hiragino Sans"
    )
    plt.rcParams["axes.unicode_minus"] = False

    window = NumericalFeedbackApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
