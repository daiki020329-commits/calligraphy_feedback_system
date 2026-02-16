"""
書道フィードバックシステム用 筆記データ収集アプリケーション

特徴:
- 複数画（ストローク）対応
- 可変長データ（リサンプリングはオプション）
- 時間情報を含む (x, y, pressure, time)
- 書道らしい太めの描画
- 入り・抜きの表現
"""

import sys
import time
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QSpinBox,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QCheckBox,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import (
    QPainter,
    QPen,
    QColor,
    QTabletEvent,
    QPainterPath,
    QFont,
    QPixmap,
)
import os
from scipy import interpolate


class CalligraphyCanvas(QWidget):
    """書道用キャンバス - 複数ストローク対応"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 700)
        self.setStyleSheet("background-color: #f0f0f0;")

        # キャンバス設定
        self.canvas_width = 600
        self.canvas_height = 600
        self.canvas_x = 0
        self.canvas_y = 0
        self.pressure_sensitivity = 1.0

        # 複数ストローク対応のデータ構造
        # strokes = [ stroke1, stroke2, ... ]
        # 各 stroke = [ [x, y, pressure, time], [x, y, pressure, time], ... ]
        self.strokes = []
        self.current_stroke = []

        # 確定済みストロークのキャッシュ画像
        self.stroke_cache = None
        self.cache_valid = False

        # 描画開始時刻（相対時間計算用）
        self.start_time = None

        # 描画状態
        self.is_drawing = False

        # タブレット入力を有効化
        self.setTabletTracking(True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)

        # 最後の描画点
        self.last_point = QPoint()

        # 描画スタイル設定
        self.min_width = 1.5  # 最小線幅（より細く）
        self.max_width = 30.0  # 最大線幅（より太く）
        self.ink_color = QColor(20, 20, 20)  # 墨色（真っ黒ではなく少しグレー）

    def get_canvas_rect(self):
        """キャンバスの矩形を取得（中央揃え）"""
        widget_width = self.width()
        widget_height = self.height()

        self.canvas_x = (widget_width - self.canvas_width) // 2
        self.canvas_y = (widget_height - self.canvas_height) // 2

        return self.canvas_x, self.canvas_y, self.canvas_width, self.canvas_height

    def is_in_canvas(self, point):
        """点がキャンバス内にあるかチェック"""
        canvas_x, canvas_y, canvas_w, canvas_h = self.get_canvas_rect()
        return (
            canvas_x <= point.x() <= canvas_x + canvas_w
            and canvas_y <= point.y() <= canvas_y + canvas_h
        )

    def point_to_normalized(self, point):
        """ウィジェット座標を正規化座標（0-1）に変換"""
        canvas_x, canvas_y, canvas_w, canvas_h = self.get_canvas_rect()

        rel_x = point.x() - canvas_x
        rel_y = point.y() - canvas_y

        norm_x = rel_x / canvas_w
        norm_y = rel_y / canvas_h

        return norm_x, norm_y

    def get_elapsed_time(self):
        """描画開始からの経過時間（ミリ秒）を取得"""
        if self.start_time is None:
            return 0
        return int((time.time() - self.start_time) * 1000)

    def tabletEvent(self, event):
        """タブレットイベントの処理"""
        if not self.is_in_canvas(event.pos()):
            if self.is_drawing:
                self.finish_current_stroke()
            return

        if event.type() == QTabletEvent.TabletPress:
            self.start_stroke(event.pos(), event.pressure())

        elif event.type() == QTabletEvent.TabletMove and self.is_drawing:
            pressure = event.pressure() * self.pressure_sensitivity
            pressure = min(1.0, max(0.0, pressure))
            self.add_point(event.pos(), pressure)

        elif event.type() == QTabletEvent.TabletRelease:
            self.finish_current_stroke()

        event.accept()

    def mousePressEvent(self, event):
        """マウス入力のフォールバック"""
        if event.button() == Qt.LeftButton and self.is_in_canvas(event.pos()):
            self.start_stroke(event.pos(), 0.5)

    def mouseMoveEvent(self, event):
        """マウス移動のフォールバック"""
        if event.buttons() & Qt.LeftButton and self.is_drawing:
            if not self.is_in_canvas(event.pos()):
                self.finish_current_stroke()
                return
            self.add_point(event.pos(), 0.5)

    def mouseReleaseEvent(self, event):
        """マウスリリースのフォールバック"""
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.finish_current_stroke()

    def start_stroke(self, pos, pressure):
        """ストロークを開始"""
        if self.start_time is None:
            self.start_time = time.time()

        self.is_drawing = True
        self.last_point = pos
        self.current_stroke = []

        # 最初の点を追加
        norm_x, norm_y = self.point_to_normalized(pos)
        elapsed = self.get_elapsed_time()
        pressure = min(1.0, max(0.0, pressure * self.pressure_sensitivity))
        self.current_stroke.append([norm_x, norm_y, pressure, elapsed])

        self.update()

    def add_point(self, pos, pressure):
        """ストロークに点を追加"""
        norm_x, norm_y = self.point_to_normalized(pos)
        elapsed = self.get_elapsed_time()
        self.current_stroke.append([norm_x, norm_y, pressure, elapsed])
        self.last_point = pos
        self.update()

    def finish_current_stroke(self):
        """現在のストロークを完了"""
        self.is_drawing = False
        if len(self.current_stroke) >= 2:  # 2点以上あれば有効なストローク
            self.strokes.append(self.current_stroke.copy())
            self.cache_valid = False  # キャッシュを無効化
        self.current_stroke = []
        self.update()

    def undo_last_stroke(self):
        """最後のストロークを削除"""
        if self.strokes:
            self.strokes.pop()
            self.cache_valid = False  # キャッシュを無効化
            self.update()
            return True
        return False

    def clear_canvas(self):
        """キャンバスをクリア"""
        self.strokes = []
        self.current_stroke = []
        self.start_time = None
        self.cache_valid = False  # キャッシュを無効化
        self.stroke_cache = None
        self.update()

    def get_stroke_count(self):
        """ストローク数を取得"""
        return len(self.strokes)

    def get_total_points(self):
        """全ストロークの総点数を取得"""
        total = sum(len(stroke) for stroke in self.strokes)
        total += len(self.current_stroke)
        return total

    def get_all_data(self):
        """全ストロークデータを取得"""
        return {
            "strokes": self.strokes.copy(),
            "stroke_count": len(self.strokes),
            "canvas_size": [self.canvas_width, self.canvas_height],
        }

    def paintEvent(self, event):
        """描画イベント"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        canvas_x, canvas_y, canvas_w, canvas_h = self.get_canvas_rect()

        # キャンバス背景（和紙風のクリーム色）
        paper_color = QColor(252, 250, 245)
        painter.fillRect(canvas_x, canvas_y, canvas_w, canvas_h, paper_color)

        # キャンバス枠線
        painter.setPen(QPen(QColor(180, 170, 160), 2))
        painter.drawRect(canvas_x, canvas_y, canvas_w, canvas_h)

        # キャンバス外をグレーアウト
        gray = QColor(240, 240, 240)
        painter.fillRect(0, 0, self.width(), canvas_y, gray)
        painter.fillRect(
            0,
            canvas_y + canvas_h,
            self.width(),
            self.height() - canvas_y - canvas_h,
            gray,
        )
        painter.fillRect(0, canvas_y, canvas_x, canvas_h, gray)
        painter.fillRect(
            canvas_x + canvas_w,
            canvas_y,
            self.width() - canvas_x - canvas_w,
            canvas_h,
            gray,
        )

        # 確定済みストロークをキャッシュから描画（または再生成）
        if self.strokes:
            if not self.cache_valid or self.stroke_cache is None:
                # キャッシュを再生成
                from PyQt5.QtGui import QPixmap

                self.stroke_cache = QPixmap(self.width(), self.height())
                self.stroke_cache.fill(Qt.transparent)
                cache_painter = QPainter(self.stroke_cache)
                cache_painter.setRenderHint(QPainter.Antialiasing)

                for stroke in self.strokes:
                    self.draw_calligraphy_stroke(cache_painter, stroke)

                cache_painter.end()
                self.cache_valid = True

            # キャッシュを描画
            painter.drawPixmap(0, 0, self.stroke_cache)

        # 現在描画中のストロークを描画（これは毎フレーム更新）
        if self.current_stroke and len(self.current_stroke) >= 2:
            self.draw_calligraphy_stroke(painter, self.current_stroke)

        # ストローク数を表示
        painter.setPen(QPen(QColor(100, 100, 100)))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            canvas_x + 5,
            canvas_y - 8,
            f"画数: {self.get_stroke_count()} | 点数: {self.get_total_points()}",
        )

    def draw_calligraphy_stroke(self, painter, stroke_data, is_finalized=True):
        """書道風のストロークを描画"""
        if len(stroke_data) < 2:
            return

        canvas_x, canvas_y, canvas_w, canvas_h = self.get_canvas_rect()

        # 筆圧を平滑化（ウィンドウを小さくして反応を良くする）
        smoothed_pressures = self.smooth_pressure(stroke_data, window_size=3)

        n_points = len(stroke_data)

        for i in range(n_points - 1):
            # 点の座標と筆圧を取得
            norm_x1, norm_y1, _, _ = stroke_data[i]
            norm_x2, norm_y2, _, _ = stroke_data[i + 1]

            p1 = smoothed_pressures[i]
            p2 = smoothed_pressures[i + 1]

            # 正規化座標を実座標に変換
            x1 = canvas_x + norm_x1 * canvas_w
            y1 = canvas_y + norm_y1 * canvas_h
            x2 = canvas_x + norm_x2 * canvas_w
            y2 = canvas_y + norm_y2 * canvas_h

            # 線幅を計算（筆圧の影響を強調）
            # p^2.0 にすることで、弱い筆圧はより細く、強い筆圧はより太く
            width1 = self.min_width + (p1**2.0) * (self.max_width - self.min_width)
            width2 = self.min_width + (p2**2.0) * (self.max_width - self.min_width)

            # 透明度も筆圧で変化（弱い筆圧は薄く）
            alpha = 150 + int((p1**1.5) * 105)  # 150-255

            self.draw_brush_segment(painter, x1, y1, x2, y2, width1, width2, alpha)

    def smooth_pressure(self, stroke_data, window_size=3):
        """筆圧データを平滑化（小さいウィンドウで反応良く）"""
        if len(stroke_data) <= window_size:
            return [point[2] for point in stroke_data]

        pressures = [point[2] for point in stroke_data]
        smoothed = []

        for i in range(len(pressures)):
            start = max(0, i - window_size // 2)
            end = min(len(pressures), i + window_size // 2 + 1)
            avg_pressure = sum(pressures[start:end]) / (end - start)
            smoothed.append(avg_pressure)

        return smoothed

    def draw_brush_segment(self, painter, x1, y1, x2, y2, width1, width2, alpha):
        """筆のセグメントを描画（書道風）"""
        import math

        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 0.5:
            return

        # セグメント数（滑らかさのため）
        segments = max(3, int(length / 3))

        for i in range(segments):
            t = i / segments
            t_next = (i + 1) / segments

            curr_x = x1 + t * (x2 - x1)
            curr_y = y1 + t * (y2 - y1)
            next_x = x1 + t_next * (x2 - x1)
            next_y = y1 + t_next * (y2 - y1)

            # 幅を補間
            curr_width = width1 + t * (width2 - width1)
            next_width = width1 + t_next * (width2 - width1)
            avg_width = (curr_width + next_width) / 2

            # メインの線を描画
            pen = QPen(QColor(20, 20, 20, alpha))
            pen.setWidthF(max(1.0, avg_width))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(int(curr_x), int(curr_y), int(next_x), int(next_y))

            # 太い線の場合、エッジをソフトにする
            if avg_width > 8:
                soft_pen = QPen(QColor(30, 30, 30, alpha // 3))
                soft_pen.setWidthF(avg_width * 1.15)
                soft_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(soft_pen)
                painter.drawLine(int(curr_x), int(curr_y), int(next_x), int(next_y))


class MainWindow(QMainWindow):
    """メインウィンドウ"""

    def __init__(self, save_dir):
        super().__init__()
        self.setWindowTitle("書道フィードバックシステム - 筆記データ収集")
        self.setGeometry(100, 50, 1000, 850)
        self.save_dir = save_dir

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # === 設定グループ ===
        settings_group = QGroupBox("設定")
        settings_layout = QFormLayout(settings_group)

        # 文字選択
        self.char_combo = QComboBox()
        self.char_combo.addItems(
            [
                "永（えい）",
                "一",
                "二",
                "人",
                "大",
                "木",
                "水",
                "火",
                "あ",
                "の",
                "カスタム",
            ]
        )
        settings_layout.addRow("対象文字:", self.char_combo)

        # キャンバスサイズ設定
        canvas_size_layout = QHBoxLayout()
        self.canvas_size_spinbox = QSpinBox()
        self.canvas_size_spinbox.setMinimum(400)
        self.canvas_size_spinbox.setMaximum(800)
        self.canvas_size_spinbox.setValue(600)
        self.canvas_size_spinbox.setSuffix(" px")
        self.canvas_size_spinbox.valueChanged.connect(self.update_canvas_size)
        canvas_size_layout.addWidget(self.canvas_size_spinbox)
        settings_layout.addRow("キャンバスサイズ:", canvas_size_layout)

        # 筆圧感度設定
        self.pressure_spinbox = QSpinBox()
        self.pressure_spinbox.setMinimum(50)
        self.pressure_spinbox.setMaximum(200)
        self.pressure_spinbox.setValue(100)
        self.pressure_spinbox.setSuffix(" %")
        self.pressure_spinbox.valueChanged.connect(self.update_pressure_sensitivity)
        settings_layout.addRow("筆圧感度:", self.pressure_spinbox)

        # リサンプリング設定
        resample_layout = QHBoxLayout()
        self.resample_checkbox = QCheckBox("リサンプリングする")
        self.resample_checkbox.setChecked(False)
        resample_layout.addWidget(self.resample_checkbox)

        self.resample_spinbox = QSpinBox()
        self.resample_spinbox.setMinimum(10)
        self.resample_spinbox.setMaximum(500)
        self.resample_spinbox.setValue(100)
        self.resample_spinbox.setSuffix(" points/stroke")
        self.resample_spinbox.setEnabled(False)
        resample_layout.addWidget(self.resample_spinbox)

        self.resample_checkbox.toggled.connect(self.resample_spinbox.setEnabled)
        settings_layout.addRow("", resample_layout)

        main_layout.addWidget(settings_group)

        # === 情報ラベル ===
        info_label = QLabel(
            "📝 ペンタブレットまたはマウスで文字を書いてください。\n"
            "💡 ペンを離すと新しい画（ストローク）になります。複数画の文字に対応しています。"
        )
        info_label.setStyleSheet(
            "padding: 10px; background-color: #e8f4fd; border-radius: 5px; color: #0066cc; font-size: 12px;"
        )
        main_layout.addWidget(info_label)

        # === キャンバス ===
        self.canvas = CalligraphyCanvas()
        main_layout.addWidget(self.canvas)

        # === ボタンレイアウト ===
        button_layout = QHBoxLayout()

        # 1画戻るボタン
        undo_button = QPushButton("↩ 1画戻る")
        undo_button.setStyleSheet(
            """
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #f57c00; }
        """
        )
        undo_button.clicked.connect(self.undo_stroke)
        button_layout.addWidget(undo_button)

        # 全消去ボタン
        clear_button = QPushButton("🗑 全消去")
        clear_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """
        )
        clear_button.clicked.connect(self.clear_canvas)
        button_layout.addWidget(clear_button)

        button_layout.addStretch()

        # ステータス表示
        self.status_label = QLabel("画数: 0 | 点数: 0")
        self.status_label.setStyleSheet(
            "padding: 10px; font-size: 14px; font-weight: bold; color: #333;"
        )
        button_layout.addWidget(self.status_label)

        button_layout.addStretch()

        # 保存ボタン
        save_button = QPushButton("💾 保存")
        save_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 30px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #388E3C; }
        """
        )
        save_button.clicked.connect(self.save_data)
        button_layout.addWidget(save_button)

        main_layout.addLayout(button_layout)

        # ステータス更新タイマー
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(200)

    def update_canvas_size(self):
        """キャンバスサイズを更新"""
        size = self.canvas_size_spinbox.value()
        self.canvas.canvas_width = size
        self.canvas.canvas_height = size
        self.canvas.update()

    def update_pressure_sensitivity(self):
        """筆圧感度を更新"""
        self.canvas.pressure_sensitivity = self.pressure_spinbox.value() / 100.0

    def update_status(self):
        """ステータス表示を更新"""
        stroke_count = self.canvas.get_stroke_count()
        point_count = self.canvas.get_total_points()
        self.status_label.setText(f"画数: {stroke_count} | 点数: {point_count}")

    def undo_stroke(self):
        """最後のストロークを削除"""
        if self.canvas.undo_last_stroke():
            self.update_status()
        else:
            QMessageBox.information(self, "情報", "削除するストロークがありません。")

    def clear_canvas(self):
        """キャンバスをクリア"""
        if self.canvas.get_stroke_count() == 0:
            return

        reply = QMessageBox.question(
            self,
            "確認",
            "すべての描画を消去しますか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.canvas.clear_canvas()
            self.update_status()

    def save_data(self):
        """描画データを保存"""
        data = self.canvas.get_all_data()

        if data["stroke_count"] == 0:
            QMessageBox.warning(self, "警告", "保存するデータがありません。")
            return

        try:
            # 文字名を取得
            char_name = self.char_combo.currentText().split("（")[0]

            # リサンプリング（オプション）
            if self.resample_checkbox.isChecked():
                target_points = self.resample_spinbox.value()
                data["strokes"] = [
                    self.resample_stroke(stroke, target_points)
                    for stroke in data["strokes"]
                ]
                data["resampled"] = True
                data["points_per_stroke"] = target_points
            else:
                data["resampled"] = False

            # メタデータを追加
            data["character"] = char_name
            data["timestamp"] = time.strftime("%Y%m%d_%H%M%S")
            data["pressure_sensitivity"] = self.pressure_spinbox.value() / 100.0

            # ファイル名を生成
            idx = 0
            while True:
                filename = f"{char_name}_{data['timestamp']}_{idx:03d}.json"
                path = os.path.join(self.save_dir, filename)
                if not os.path.exists(path):
                    break
                idx += 1

            # JSON形式で保存
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # NumPy形式でも保存（互換性のため）
            npy_path = path.replace(".json", ".npy")
            # 全ストロークを結合した配列
            all_points = []
            for stroke in data["strokes"]:
                all_points.extend(stroke)
            if all_points:
                np.save(npy_path, np.array(all_points))

            # 保存完了メッセージ
            msg = (
                f"データを保存しました:\n\n"
                f"📁 {filename}\n"
                f"✏️ 文字: {char_name}\n"
                f"📊 画数: {data['stroke_count']}\n"
                f"📍 総点数: {sum(len(s) for s in data['strokes'])}\n"
            )
            if data["resampled"]:
                msg += f"🔄 リサンプル: {data['points_per_stroke']} points/stroke\n"

            QMessageBox.information(self, "保存完了", msg)

            # キャンバスをクリアするか確認
            reply = QMessageBox.question(
                self,
                "確認",
                "キャンバスをクリアして次の文字を書きますか？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.canvas.clear_canvas()

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"保存に失敗しました:\n{str(e)}")

    def resample_stroke(self, stroke, target_points):
        """ストロークを指定点数にリサンプリング"""
        if len(stroke) < 2:
            return stroke

        if len(stroke) == target_points:
            return stroke

        stroke_array = np.array(stroke)
        current_points = len(stroke_array)

        t_original = np.linspace(0, 1, current_points)
        t_target = np.linspace(0, 1, target_points)

        resampled = np.zeros((target_points, 4))  # x, y, pressure, time

        for i in range(4):
            f = interpolate.interp1d(t_original, stroke_array[:, i], kind="linear")
            resampled[:, i] = f(t_target)

        return resampled.tolist()


def main():
    # 保存先ディレクトリ
    save_dir = "calligraphy_data"
    os.makedirs(save_dir, exist_ok=True)

    app = QApplication(sys.argv)
    app.setApplicationName("書道フィードバックシステム - 筆記入力")
    app.setApplicationVersion("2.0")

    # フォント設定（日本語対応）
    font = QFont()
    font.setFamily("Hiragino Sans" if sys.platform == "darwin" else "Yu Gothic UI")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow(save_dir)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
