"""
書道フィードバック GUI アプリケーション

キャンバスに書いて送信すると、LLMによるフィードバックを受けられる。
会話形式で繰り返し練習し、前回との比較コメントも得られる。

使い方:
    python gui_LLM_feedback.py
"""

import glob
import json
import math
import os
import sys
import time
import traceback

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
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QBuffer, QIODevice, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QImage

from calligraphy_input_app import CalligraphyCanvas
from feature_extractor import extract_features
from diff_calculator import compute_diff, diff_to_text
from feedback_generator import generate_feedback_multiturn


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
# QPainter ベースのレンダリング（CalligraphyCanvas と同じ描画ロジック）
# ---------------------------------------------------------------------------

# CalligraphyCanvas と同じ描画パラメータ
_MIN_WIDTH = 1.5
_MAX_WIDTH = 30.0
_PRESSURE_EXP = 2.0
_INK_COLOR = (20, 20, 20)
_PAPER_COLOR = QColor(252, 250, 245)
_BORDER_COLOR = QColor(180, 170, 160)


def _smooth_pressure(stroke_data: list, window_size: int = 3) -> list[float]:
    """CalligraphyCanvas.smooth_pressure と同一のロジック。"""
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
    """CalligraphyCanvas.draw_brush_segment と同一のロジック。"""
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
    """CalligraphyCanvas.draw_calligraphy_stroke と同一のロジック。"""
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
    """QPixmap を PNG バイト列に変換する。"""
    ba = bytearray()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    return bytes(buf.data())


def _render_strokes_to_png(stroke_data: dict, size: int = 600) -> bytes:
    """QPainter でストロークを描画して PNG を返す。"""
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


def _create_comparison_png(
    ref_data: dict,
    user_data: dict,
    size: int = 600,
) -> bytes:
    """QPainter で Reference / User の比較画像を生成して PNG を返す。"""
    gap = 20
    label_h = 40
    total_w = size * 2 + gap
    total_h = size + label_h

    pixmap = QPixmap(total_w, total_h)
    pixmap.fill(QColor(240, 240, 240))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # ラベル
    label_font = QFont()
    label_font.setPointSize(14)
    label_font.setBold(True)
    painter.setFont(label_font)
    painter.setPen(QColor(80, 80, 80))
    painter.drawText(size // 2 - 40, 28, "Reference")
    painter.drawText(size + gap + size // 2 - 20, 28, "User")

    # お手本
    ref_pix = QPixmap(size, size)
    ref_pix.fill(_PAPER_COLOR)
    rp = QPainter(ref_pix)
    rp.setRenderHint(QPainter.Antialiasing)
    rp.setPen(QPen(_BORDER_COLOR, 2))
    rp.drawRect(0, 0, size - 1, size - 1)
    for s in ref_data.get("strokes", []):
        _draw_stroke_qp(rp, s, size)
    rp.end()
    painter.drawPixmap(0, label_h, ref_pix)

    # ユーザー
    usr_pix = QPixmap(size, size)
    usr_pix.fill(_PAPER_COLOR)
    up = QPainter(usr_pix)
    up.setRenderHint(QPainter.Antialiasing)
    up.setPen(QPen(_BORDER_COLOR, 2))
    up.drawRect(0, 0, size - 1, size - 1)
    for s in user_data.get("strokes", []):
        _draw_stroke_qp(up, s, size)
    up.end()
    painter.drawPixmap(size + gap, label_h, usr_pix)

    painter.end()
    return _pixmap_to_png(pixmap)


# ---------------------------------------------------------------------------
# FeedbackEntry: 1回分のフィードバック表示ウィジェット
# ---------------------------------------------------------------------------


class FeedbackEntry(QFrame):
    """試行番号 + 比較画像サムネイル + フィードバックテキストを表示する。"""

    def __init__(
        self,
        attempt_number: int,
        comparison_png: bytes,
        feedback_text: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "FeedbackEntry { background-color: #ffffff; border: 1px solid #ddd; "
            "border-radius: 6px; margin: 4px 0; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # --- 試行番号ヘッダー ---
        header = QLabel(f"Attempt #{attempt_number}")
        header.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #333; border: none;"
        )
        layout.addWidget(header)

        # --- 比較画像サムネイル ---
        thumb_label = QLabel()
        thumb_label.setStyleSheet("border: none;")
        qimg = QImage.fromData(comparison_png)
        if not qimg.isNull():
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaledToWidth(380, Qt.SmoothTransformation)
            thumb_label.setPixmap(scaled)
        thumb_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(thumb_label)

        # --- フィードバックテキスト ---
        text_label = QLabel(feedback_text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet(
            "font-size: 13px; color: #222; line-height: 1.5; padding: 4px 0; border: none;"
        )
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(text_label)


# ---------------------------------------------------------------------------
# FeedbackWorker: バックグラウンドでAPI呼び出し
# ---------------------------------------------------------------------------


class FeedbackWorker(QThread):
    """特徴量抽出 → 差分計算 → API呼び出しをバックグラウンドで実行する。

    比較画像は QPainter で生成する必要があるため、メインスレッドで事前に生成し
    コンストラクタに渡す。
    """

    finished = pyqtSignal(
        bytes, str, list
    )  # comparison_png, feedback_text, updated_history
    error = pyqtSignal(str)
    progress = pyqtSignal(str)  # ステータスメッセージ

    def __init__(
        self,
        ref_data: dict,
        user_data: dict,
        character: str,
        conversation_history: list[dict],
        attempt_number: int,
        comparison_png: bytes,
    ):
        super().__init__()
        self.ref_data = ref_data
        self.user_data = user_data
        self.character = character
        self.conversation_history = conversation_history
        self.attempt_number = attempt_number
        self.comparison_png = comparison_png

    def run(self):
        try:
            # 1. 特徴量抽出
            self.progress.emit("特徴量を抽出中...")
            ref_features = extract_features(self.ref_data)
            user_features = extract_features(self.user_data)

            # 2. 差分テキスト
            self.progress.emit("差分を計算中...")
            diffs = compute_diff(ref_features, user_features)
            body_text = diff_to_text(diffs)

            # 3. LLMフィードバック
            self.progress.emit("フィードバックを生成中...")
            updated_history, feedback_text = generate_feedback_multiturn(
                comparison_image=self.comparison_png,
                body_data_text=body_text,
                character=self.character,
                conversation_history=self.conversation_history,
                attempt_number=self.attempt_number,
            )

            self.finished.emit(self.comparison_png, feedback_text, updated_history)

        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# CalligraphyFeedbackApp: メインウィンドウ
# ---------------------------------------------------------------------------


class CalligraphyFeedbackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("書道フィードバック")
        self.setGeometry(80, 40, 1200, 800)

        # --- 状態 ---
        self.reference_data: dict | None = None
        self.conversation_history: list[dict] = []
        self.attempt_number: int = 0
        self.worker: FeedbackWorker | None = None

        # 参照データマップ {表示名: filepath}
        self._ref_map: dict[str, str] = {}

        self._build_ui()
        self._scan_references()

    # ---------------------------------------------------------------
    # UI構築
    # ---------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 0)
        root_layout.setSpacing(6)

        # --- 上部: 参照文字選択 ---
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("参照文字:"))
        self.char_combo = QComboBox()
        self.char_combo.setMinimumWidth(200)
        self.char_combo.currentIndexChanged.connect(self._on_reference_changed)
        top_bar.addWidget(self.char_combo)
        top_bar.addStretch()
        root_layout.addLayout(top_bar)

        # --- メイン: 左右スプリッター ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # -- 左パネル --
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # お手本画像
        self.ref_image_label = QLabel("お手本画像")
        self.ref_image_label.setFixedSize(300, 300)
        self.ref_image_label.setAlignment(Qt.AlignCenter)
        self.ref_image_label.setStyleSheet(
            "background-color: #fcfaf5; border: 1px solid #bbb; border-radius: 4px;"
        )
        left_layout.addWidget(self.ref_image_label, 0, Qt.AlignHCenter)

        # キャンバス
        self.canvas = CalligraphyCanvas()
        self.canvas.setMinimumSize(600, 600)
        self.canvas.setMaximumSize(600, 600)
        left_layout.addWidget(self.canvas, 0, Qt.AlignHCenter)

        # ボタン行
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

        # -- 右パネル (スクロール可能なフィードバック履歴) --
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
        self.feedback_layout.addStretch()  # 下部にスペーサー

        self.scroll_area.setWidget(self.feedback_container)
        right_layout.addWidget(self.scroll_area)

        splitter.addWidget(right_panel)

        # スプリッター比率
        splitter.setStretchFactor(0, 0)  # 左は固定幅寄り
        splitter.setStretchFactor(1, 1)  # 右は伸縮
        splitter.setSizes([640, 500])

        root_layout.addWidget(splitter, 1)

        # --- ステータスバー ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了")

    # ---------------------------------------------------------------
    # 参照データ管理
    # ---------------------------------------------------------------

    def _scan_references(self):
        """reference_data/*.json をスキャンしてComboBoxに追加する。"""
        ref_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reference_data"
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

        # 最初の参照データを読み込む
        if self.char_combo.count() > 0:
            self._on_reference_changed(0)

    def _on_reference_changed(self, index: int):
        """参照文字が変わったら状態をリセットする。"""
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

        # 会話リセット
        self.conversation_history = []
        self.attempt_number = 0
        self._clear_feedback_entries()
        self.canvas.clear_canvas()

        # お手本画像を更新
        self._update_ref_image()
        self.status_bar.showMessage(
            f"参照文字「{self.reference_data.get('character', '?')}」を読み込みました"
        )

    def _update_ref_image(self):
        """お手本のストロークを画像化して表示する（QPainter ベース）。"""
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
    # 送信 → フィードバック
    # ---------------------------------------------------------------

    def _on_submit(self):
        """送信ボタン押下: バックグラウンドでパイプライン実行。"""
        if self.reference_data is None:
            QMessageBox.warning(self, "警告", "参照データが選択されていません。")
            return

        user_data = self.canvas.get_all_data()
        if user_data["stroke_count"] == 0:
            QMessageBox.warning(
                self, "警告", "キャンバスに何か書いてから送信してください。"
            )
            return

        if not os.environ.get("ANTHROPIC_API_KEY"):
            QMessageBox.warning(
                self,
                "APIキー未設定",
                "環境変数 ANTHROPIC_API_KEY が設定されていません。\n"
                "set ANTHROPIC_API_KEY=your-key で設定してください。",
            )
            return

        # ボタン無効化
        self.submit_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        self.attempt_number += 1
        character = self.reference_data.get("character", "?")

        # 比較画像をメインスレッドで生成（QPainter はメインスレッド限定）
        self.status_bar.showMessage("比較画像を生成中...")
        comparison_png = _create_comparison_png(self.reference_data, user_data)

        self.worker = FeedbackWorker(
            ref_data=self.reference_data,
            user_data=user_data,
            character=character,
            conversation_history=list(self.conversation_history),
            attempt_number=self.attempt_number,
            comparison_png=comparison_png,
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, message: str):
        self.status_bar.showMessage(message)

    def _on_worker_finished(
        self, comparison_png: bytes, feedback_text: str, updated_history: list
    ):
        self.conversation_history = updated_history

        # 自動保存（キャンバスクリア前に実行）
        saved_path = self._save_attempt(self.worker.user_data, feedback_text)

        # フィードバックエントリを追加
        entry = FeedbackEntry(self.attempt_number, comparison_png, feedback_text)
        # ストレッチの前に挿入
        idx = self.feedback_layout.count() - 1  # 末尾のストレッチの前
        self.feedback_layout.insertWidget(idx, entry)

        # 最下部にスクロール
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        # 少し待ってから再度スクロール（レイアウト更新後に確実に最下部へ）
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(
            100,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

        # キャンバスクリア & ボタン復元
        self.canvas.clear_canvas()
        self.submit_btn.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        status = f"Attempt #{self.attempt_number} のフィードバックを受信しました"
        if saved_path:
            status += f" (保存: {os.path.basename(saved_path)})"
        self.status_bar.showMessage(status)

        self.worker = None

    def _on_worker_error(self, error_text: str):
        self.attempt_number -= 1  # 失敗したので戻す
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

    def _save_attempt(self, user_data: dict, feedback_text: str) -> str | None:
        """ユーザー入力とフィードバックを user_data/ に自動保存する。

        Returns:
            保存したファイルパス。失敗時は None。
        """
        try:
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "user_data",
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
                "feedback": feedback_text,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            return path
        except Exception as e:
            self.status_bar.showMessage(f"保存失敗: {e}")
            return None

    def _clear_feedback_entries(self):
        """右パネルのフィードバックエントリを全削除する。"""
        while self.feedback_layout.count() > 1:  # ストレッチを残す
            item = self.feedback_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("書道フィードバック")

    font = QFont()
    font.setFamily("Yu Gothic UI" if sys.platform == "win32" else "Hiragino Sans")
    font.setPointSize(10)
    app.setFont(font)

    window = CalligraphyFeedbackApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
