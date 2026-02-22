"""
バッチスコアリングモジュール

ユーザーの書道データをお手本と比較し、DTW類似度とSSIM類似度の
複合スコアを算出してCSVに出力する。

使い方:
    python experiment/scoring.py --ref-dir reference_data --user-dir experiment/user_data_no_feedback --output scores_A.csv
    python experiment/scoring.py --ref-dir reference_data --user-dir experiment/user_data_image_only --output scores_B.csv
    python experiment/scoring.py --ref-dir reference_data --user-dir user_data --output scores_C.csv
"""

import argparse
import csv
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from dtw import dtw
from skimage.metrics import structural_similarity

# 親ディレクトリをパスに追加（image_generator を使うため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_generator import render_stroke_to_image


# ---------------------------------------------------------------------------
# DTW 類似度
# ---------------------------------------------------------------------------

def _extract_xy_sequences(stroke_data: dict) -> list[np.ndarray]:
    """ストロークデータから各ストロークの (x, y) 座標列を抽出する。"""
    sequences = []
    for stroke in stroke_data.get("strokes", []):
        if len(stroke) < 2:
            continue
        coords = np.array([[pt[0], pt[1]] for pt in stroke], dtype=float)
        sequences.append(coords)
    return sequences


def _extract_pressure_sequences(stroke_data: dict) -> list[np.ndarray]:
    """ストロークデータから各ストロークの筆圧列を抽出する。"""
    sequences = []
    for stroke in stroke_data.get("strokes", []):
        if len(stroke) < 2:
            continue
        pressures = np.array([[pt[2]] for pt in stroke], dtype=float)
        sequences.append(pressures)
    return sequences


def compute_dtw_similarity(ref_data: dict, user_data: dict) -> float:
    """ストロークごとの DTW 距離の平均から類似度を算出する。

    類似度 = 1 / (1 + mean_distance)

    ストローク数が異なる場合、少ない方に合わせてペアリングする。
    余ったストロークはペナルティとして最大距離を加算する。

    Returns:
        (0, 1] の範囲の類似度スコア
    """
    ref_seqs = _extract_xy_sequences(ref_data)
    user_seqs = _extract_xy_sequences(user_data)

    if not ref_seqs or not user_seqs:
        return 0.0

    distances = []
    n_pairs = min(len(ref_seqs), len(user_seqs))

    for i in range(n_pairs):
        alignment = dtw(ref_seqs[i], user_seqs[i])
        distances.append(alignment.distance)

    # ストローク数の差にペナルティ
    extra = abs(len(ref_seqs) - len(user_seqs))
    if extra > 0:
        penalty = max(distances) if distances else 1.0
        distances.extend([penalty] * extra)

    mean_dist = np.mean(distances)
    return 1.0 / (1.0 + mean_dist)


def compute_dtw_pressure_similarity(ref_data: dict, user_data: dict) -> float:
    """ストロークごとの筆圧列の DTW 距離の平均から類似度を算出する。

    類似度 = 1 / (1 + mean_distance)

    Returns:
        (0, 1] の範囲の類似度スコア
    """
    ref_seqs = _extract_pressure_sequences(ref_data)
    user_seqs = _extract_pressure_sequences(user_data)

    if not ref_seqs or not user_seqs:
        return 0.0

    distances = []
    n_pairs = min(len(ref_seqs), len(user_seqs))

    for i in range(n_pairs):
        alignment = dtw(ref_seqs[i], user_seqs[i])
        distances.append(alignment.distance)

    extra = abs(len(ref_seqs) - len(user_seqs))
    if extra > 0:
        penalty = max(distances) if distances else 1.0
        distances.extend([penalty] * extra)

    mean_dist = np.mean(distances)
    return 1.0 / (1.0 + mean_dist)


# ---------------------------------------------------------------------------
# SSIM 類似度
# ---------------------------------------------------------------------------

def _render_to_grayscale(stroke_data: dict, size: int = 600) -> np.ndarray:
    """ストロークデータを PIL でレンダリングし、グレースケール numpy 配列に変換する。"""
    png_bytes = render_stroke_to_image(stroke_data, size=size)
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    return np.array(img)


def compute_ssim_similarity(ref_data: dict, user_data: dict, size: int = 600) -> float:
    """SSIM（構造的類似性指標）を算出する。

    Returns:
        [0, 1] の範囲の類似度スコア
    """
    ref_gray = _render_to_grayscale(ref_data, size)
    user_gray = _render_to_grayscale(user_data, size)
    score = structural_similarity(ref_gray, user_gray)
    return max(0.0, score)


# ---------------------------------------------------------------------------
# 複合スコア
# ---------------------------------------------------------------------------

def compute_composite_score(
    ref_data: dict,
    user_data: dict,
    dtw_weight: float = 0.5,
    ssim_weight: float = 0.5,
) -> dict:
    """DTW + SSIM の複合スコアを算出する。

    Returns:
        {
            "dtw_similarity": float,
            "ssim_similarity": float,
            "composite_score": float,
        }
    """
    dtw_sim = compute_dtw_similarity(ref_data, user_data)
    ssim_sim = compute_ssim_similarity(ref_data, user_data)
    composite = dtw_weight * dtw_sim + ssim_weight * ssim_sim

    return {
        "dtw_similarity": round(dtw_sim, 6),
        "ssim_similarity": round(ssim_sim, 6),
        "composite_score": round(composite, 6),
    }


def compute_composite_score_v2(
    ref_data: dict,
    user_data: dict,
    dtw_xy_weight: float = 0.4,
    dtw_pressure_weight: float = 0.2,
    ssim_weight: float = 0.4,
) -> dict:
    """DTW座標 + DTW筆圧 + SSIM の複合スコアを算出する。

    Returns:
        {
            "dtw_xy_similarity": float,
            "dtw_pressure_similarity": float,
            "ssim_similarity": float,
            "composite_score": float,
        }
    """
    dtw_xy = compute_dtw_similarity(ref_data, user_data)
    dtw_pressure = compute_dtw_pressure_similarity(ref_data, user_data)
    ssim_sim = compute_ssim_similarity(ref_data, user_data)
    composite = dtw_xy_weight * dtw_xy + dtw_pressure_weight * dtw_pressure + ssim_weight * ssim_sim

    return {
        "dtw_xy_similarity": round(dtw_xy, 6),
        "dtw_pressure_similarity": round(dtw_pressure, 6),
        "ssim_similarity": round(ssim_sim, 6),
        "composite_score": round(composite, 6),
    }


# ---------------------------------------------------------------------------
# リファレンスデータのマッチング
# ---------------------------------------------------------------------------

def _load_reference_map(ref_dir: str) -> dict[str, dict]:
    """参照ディレクトリから {文字名: データ} のマップを構築する。"""
    ref_map = {}
    for fname in os.listdir(ref_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ref_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        character = data.get("character", os.path.splitext(fname)[0])
        ref_map[character] = data
    return ref_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="書道データのバッチスコアリング",
    )
    parser.add_argument(
        "--ref-dir",
        required=True,
        help="参照データのディレクトリ (例: reference_data)",
    )
    parser.add_argument(
        "--user-dir",
        required=True,
        help="ユーザーデータのディレクトリ (例: experiment/user_data_no_feedback)",
    )
    parser.add_argument(
        "--output",
        default="scores.csv",
        help="出力CSVファイル名 (デフォルト: scores.csv)",
    )
    args = parser.parse_args()

    # パスをプロジェクトルートからの相対パスとして解決
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = os.path.join(project_root, args.ref_dir) if not os.path.isabs(args.ref_dir) else args.ref_dir
    user_dir = os.path.join(project_root, args.user_dir) if not os.path.isabs(args.user_dir) else args.user_dir

    if not os.path.isdir(ref_dir):
        print(f"エラー: 参照ディレクトリが見つかりません: {ref_dir}")
        sys.exit(1)

    if not os.path.isdir(user_dir):
        print(f"エラー: ユーザーディレクトリが見つかりません: {user_dir}")
        sys.exit(1)

    # 参照データ読み込み
    ref_map = _load_reference_map(ref_dir)
    if not ref_map:
        print(f"エラー: 参照データが見つかりません: {ref_dir}")
        sys.exit(1)

    print(f"参照データ: {list(ref_map.keys())}")

    # ユーザーデータを処理
    user_files = sorted(f for f in os.listdir(user_dir) if f.endswith(".json"))
    if not user_files:
        print(f"エラー: ユーザーデータが見つかりません: {user_dir}")
        sys.exit(1)

    print(f"ユーザーデータ: {len(user_files)} ファイル")

    results = []
    for fname in user_files:
        path = os.path.join(user_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        character = user_data.get("character", "")
        attempt_number = user_data.get("attempt_number", 0)

        if character not in ref_map:
            print(f"  スキップ: {fname} (参照データなし: '{character}')")
            continue

        print(f"  スコアリング: {fname} ...", end=" ", flush=True)
        scores = compute_composite_score_v2(ref_map[character], user_data)
        print(
            f"DTW_xy={scores['dtw_xy_similarity']:.4f} "
            f"DTW_pressure={scores['dtw_pressure_similarity']:.4f} "
            f"SSIM={scores['ssim_similarity']:.4f} "
            f"Composite={scores['composite_score']:.4f}"
        )

        results.append({
            "filename": fname,
            "character": character,
            "attempt_number": attempt_number,
            **scores,
        })

    # CSV 出力
    output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
    fieldnames = [
        "filename",
        "character",
        "attempt_number",
        "dtw_xy_similarity",
        "dtw_pressure_similarity",
        "ssim_similarity",
        "composite_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n結果を保存しました: {output_path} ({len(results)} 件)")


if __name__ == "__main__":
    main()
