"""
条件A: フィードバック無し GUI

お手本を見て練習するだけ。フィードバックは表示しない。
保存先: experiment/user_data_no_feedback/

使い方:
    python experiment/gui_no_feedback.py
"""

import glob
import json
import math
import os
import sys
import time

# 親ディレクトリをパスに追加して CalligraphyCanvas をインポート
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QBuffer, QIODevice
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QImage

from calligraphy_input_app import CalligraphyCanvas


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
# NoFeedbackApp: メインウィンドウ
# ---------------------------------------------------------------------------

class NoFeedbackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("書道練習（条件A: フィードバック無し）")
        self.setGeometry(80, 40, 700, 1000)

        self.reference_data: dict | None = None
        self.attempt_number: int = 0
        self._ref_map: dict[str, str] = {}

        self._build_ui()
        self._scan_references()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(6)

        # 参照文字選択
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("参照文字:"))
        self.char_combo = QComboBox()
        self.char_combo.setMinimumWidth(200)
        self.char_combo.currentIndexChanged.connect(self._on_reference_changed)
        top_bar.addWidget(self.char_combo)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # お手本画像
        self.ref_image_label = QLabel("お手本画像")
        self.ref_image_label.setFixedSize(300, 300)
        self.ref_image_label.setAlignment(Qt.AlignCenter)
        self.ref_image_label.setStyleSheet(
            "background-color: #fcfaf5; border: 1px solid #bbb; border-radius: 4px;"
        )
        layout.addWidget(self.ref_image_label, 0, Qt.AlignHCenter)

        # キャンバス
        self.canvas = CalligraphyCanvas()
        self.canvas.setMinimumSize(600, 600)
        self.canvas.setMaximumSize(600, 600)
        layout.addWidget(self.canvas, 0, Qt.AlignHCenter)

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

        self.save_btn = QPushButton("保存")
        self.save_btn.setStyleSheet(BUTTON_STYLE_GREEN)
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)

        layout.addLayout(btn_row)
        layout.addStretch()

        # ステータスバー
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
                    300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation,
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
    # 保存
    # ---------------------------------------------------------------

    def _on_save(self):
        if self.reference_data is None:
            QMessageBox.warning(self, "警告", "参照データが選択されていません。")
            return

        user_data = self.canvas.get_all_data()
        if user_data["stroke_count"] == 0:
            QMessageBox.warning(self, "警告", "キャンバスに何か書いてから保存してください。")
            return

        self.attempt_number += 1
        saved_path = self._save_attempt(user_data)

        if saved_path:
            self.status_bar.showMessage(
                f"Attempt #{self.attempt_number} を保存しました ({os.path.basename(saved_path)})"
            )
        else:
            self.attempt_number -= 1

        self.canvas.clear_canvas()

    def _save_attempt(self, user_data: dict) -> str | None:
        try:
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "user_data_no_feedback",
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
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            return path
        except Exception as e:
            self.status_bar.showMessage(f"保存失敗: {e}")
            return None


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("書道練習（条件A）")

    font = QFont()
    font.setFamily("Yu Gothic UI" if sys.platform == "win32" else "Hiragino Sans")
    font.setPointSize(10)
    app.setFont(font)

    window = NoFeedbackApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
