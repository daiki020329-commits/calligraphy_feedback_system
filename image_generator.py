"""
画像生成モジュール

筆記データを画像化する。
- 筆圧を線の太さに反映 (p^2.0)
- 最小線幅: 1.5px, 最大線幅: 30px
- 背景: 和紙風クリーム色 RGB(252, 250, 245)
"""

import io
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 描画定数
def _get_japanese_font(size: int = 20) -> ImageFont.FreeTypeFont | None:
    """日本語対応フォントを取得する。"""
    # Windows
    if sys.platform == "win32":
        candidates = [
            "C:/Windows/Fonts/yugothib.ttf",   # Yu Gothic Bold
            "C:/Windows/Fonts/yugothic.ttf",    # Yu Gothic
            "C:/Windows/Fonts/meiryo.ttc",      # Meiryo
            "C:/Windows/Fonts/msgothic.ttc",    # MS Gothic
        ]
    # macOS
    elif sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    # Linux
    else:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return None


CANVAS_SIZE = 600
BACKGROUND_COLOR = (252, 250, 245)
INK_COLOR = (20, 20, 20)
MIN_WIDTH = 1.5
MAX_WIDTH = 30.0
PRESSURE_EXPONENT = 2.0
PADDING = 40  # キャンバス内のパディング


def render_stroke_to_image(stroke_data: dict, size: int = CANVAS_SIZE) -> bytes:
    """筆記データを画像化してPNGバイト列を返す。

    Args:
        stroke_data: {"strokes": [[[x, y, pressure, time], ...], ...]}
        size: 画像サイズ（正方形）

    Returns:
        PNG画像のバイト列
    """
    img = Image.new("RGB", (size, size), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # 枠線を描画
    _draw_border(draw, size)

    strokes = stroke_data.get("strokes", [])
    for stroke in strokes:
        _draw_stroke(draw, stroke, size)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_comparison_image(
    ref_stroke_data: dict,
    user_stroke_data: dict,
    size: int = CANVAS_SIZE,
) -> bytes:
    """お手本とユーザーの比較画像を生成する。

    横に並べた比較画像を返す。

    Args:
        ref_stroke_data: お手本の筆記データ
        user_stroke_data: ユーザーの筆記データ
        size: 各画像のサイズ

    Returns:
        PNG画像のバイト列
    """
    gap = 20
    label_height = 40
    total_width = size * 2 + gap
    total_height = size + label_height

    comparison = Image.new("RGB", (total_width, total_height), (240, 240, 240))
    draw_comp = ImageDraw.Draw(comparison)

    # ラベル描画
    draw_comp.text((size // 2 - 40, 10), "Reference", fill=(80, 80, 80))
    draw_comp.text((size + gap + size // 2 - 20, 10), "User", fill=(80, 80, 80))

    # お手本画像
    ref_img = Image.new("RGB", (size, size), BACKGROUND_COLOR)
    ref_draw = ImageDraw.Draw(ref_img)
    _draw_border(ref_draw, size)
    for stroke in ref_stroke_data.get("strokes", []):
        _draw_stroke(ref_draw, stroke, size)
    comparison.paste(ref_img, (0, label_height))

    # ユーザー画像
    user_img = Image.new("RGB", (size, size), BACKGROUND_COLOR)
    user_draw = ImageDraw.Draw(user_img)
    _draw_border(user_draw, size)
    for stroke in user_stroke_data.get("strokes", []):
        _draw_stroke(user_draw, stroke, size)
    comparison.paste(user_img, (size + gap, label_height))

    buf = io.BytesIO()
    comparison.save(buf, format="PNG")
    return buf.getvalue()


def _draw_border(draw: ImageDraw.Draw, size: int):
    """キャンバスの枠線を描画する。"""
    border_color = (180, 170, 160)
    draw.rectangle([0, 0, size - 1, size - 1], outline=border_color, width=2)


def _draw_stroke(draw: ImageDraw.Draw, stroke: list, size: int):
    """1つのストロークを描画する。

    筆圧を線の太さに反映し、書道らしい表現をする。
    """
    if len(stroke) < 2:
        return

    points = np.array(stroke, dtype=float)

    for i in range(len(points) - 1):
        x1 = points[i][0] * size
        y1 = points[i][1] * size
        x2 = points[i + 1][0] * size
        y2 = points[i + 1][1] * size
        p1 = points[i][2]
        p2 = points[i + 1][2]

        # 筆圧から線幅を計算 (p^2.0)
        width1 = MIN_WIDTH + (p1 ** PRESSURE_EXPONENT) * (MAX_WIDTH - MIN_WIDTH)
        width2 = MIN_WIDTH + (p2 ** PRESSURE_EXPONENT) * (MAX_WIDTH - MIN_WIDTH)

        seg_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if seg_len < 0.5:
            continue

        # セグメントを細かく分割して滑らかに描画
        n_sub = max(3, int(seg_len / 3))
        for j in range(n_sub):
            t1 = j / n_sub
            t2 = (j + 1) / n_sub

            sx1 = x1 + t1 * (x2 - x1)
            sy1 = y1 + t1 * (y2 - y1)
            sx2 = x1 + t2 * (x2 - x1)
            sy2 = y1 + t2 * (y2 - y1)

            w = width1 + ((t1 + t2) / 2) * (width2 - width1)

            # 透明度の代わりに色の濃さで筆圧を表現
            avg_p = p1 + ((t1 + t2) / 2) * (p2 - p1)
            alpha_factor = 0.6 + 0.4 * (avg_p ** 1.5)
            ink = tuple(int(c * (1 - alpha_factor) + 255 * (alpha_factor - 1) + c) for c in INK_COLOR)
            ink = tuple(max(0, min(255, int(20 + (235 - 20) * (1 - alpha_factor)))) for _ in range(3))

            draw.line(
                [(sx1, sy1), (sx2, sy2)],
                fill=ink,
                width=max(1, int(w)),
            )

            # 太い線の接合部を円で埋める（滑らかに見せる）
            if w > 3:
                r = w / 2
                draw.ellipse(
                    [sx1 - r, sy1 - r, sx1 + r, sy1 + r],
                    fill=ink,
                )

    # 最後の点にも円を描画
    last = points[-1]
    lx, ly = last[0] * size, last[1] * size
    lp = last[2]
    lw = MIN_WIDTH + (lp ** PRESSURE_EXPONENT) * (MAX_WIDTH - MIN_WIDTH)
    if lw > 3:
        r = lw / 2
        ink_last = tuple(max(0, min(255, int(20 + 215 * (1 - (0.6 + 0.4 * (lp ** 1.5)))))) for _ in range(3))
        draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=ink_last)
