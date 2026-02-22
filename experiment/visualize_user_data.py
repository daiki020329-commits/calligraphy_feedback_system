"""
ユーザーデータ可視化スクリプト

保存済みの JSON ファイルからキャンバスと同一の描画ロジックで画像を生成する。

使い方:
    # 単一ファイル
    python experiment/visualize_user_data.py path/to/user_data.json

    # ディレクトリ内の全ファイル
    python experiment/visualize_user_data.py path/to/user_data_dir/

    # 出力先を指定
    python experiment/visualize_user_data.py path/to/data --output-dir output_images

    # 画像サイズを指定
    python experiment/visualize_user_data.py path/to/data --size 800
"""

import argparse
import json
import math
import os
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QBuffer, QIODevice
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap


# ---------------------------------------------------------------------------
# 描画パラメータ（CalligraphyCanvas / gui_app.py と同一）
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
    x1: float, y1: float, x2: float, y2: float,
    width1: float, width2: float, alpha: int,
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
        w1 = _MIN_WIDTH + (p1 ** _PRESSURE_EXP) * (_MAX_WIDTH - _MIN_WIDTH)
        w2 = _MIN_WIDTH + (p2 ** _PRESSURE_EXP) * (_MAX_WIDTH - _MIN_WIDTH)
        alpha = 150 + int((p1 ** 1.5) * 105)
        _draw_brush_segment_qp(painter, x1, y1, x2, y2, w1, w2, alpha)


def render_strokes_to_png(stroke_data: dict, size: int = 600) -> bytes:
    """ストロークデータを QPainter で描画して PNG バイト列を返す。"""
    pixmap = QPixmap(size, size)
    pixmap.fill(_PAPER_COLOR)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(_BORDER_COLOR, 2))
    painter.drawRect(0, 0, size - 1, size - 1)
    for stroke in stroke_data.get("strokes", []):
        _draw_stroke_qp(painter, stroke, size)
    painter.end()

    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    return bytes(buf.data())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ユーザーデータの筆記画像を可視化する",
    )
    parser.add_argument(
        "path",
        help="JSONファイルまたはJSONファイルを含むディレクトリのパス",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="画像の出力先ディレクトリ（デフォルト: 入力と同じディレクトリ）",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="画像サイズ（正方形、デフォルト: 600）",
    )
    args = parser.parse_args()

    # QApplication が必要（QPainter / QPixmap のため）
    app = QApplication(sys.argv)

    # 入力パスの解決
    input_path = os.path.abspath(args.path)

    if os.path.isfile(input_path):
        json_files = [input_path]
    elif os.path.isdir(input_path):
        json_files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        )
    else:
        print(f"エラー: パスが見つかりません: {input_path}")
        sys.exit(1)

    if not json_files:
        print(f"エラー: JSONファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 出力先の決定
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    elif os.path.isfile(input_path):
        output_dir = os.path.dirname(input_path)
    else:
        output_dir = input_path
    os.makedirs(output_dir, exist_ok=True)

    print(f"{len(json_files)} ファイルを処理します → {output_dir}")

    for json_path in json_files:
        basename = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join(output_dir, f"{basename}.png")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        png_bytes = render_strokes_to_png(data, size=args.size)

        with open(out_path, "wb") as f:
            f.write(png_bytes)

        print(f"  {os.path.basename(json_path)} → {os.path.basename(out_path)}")

    print("完了")


if __name__ == "__main__":
    main()
