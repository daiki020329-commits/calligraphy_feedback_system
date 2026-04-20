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


def _extract_speed_sequences(stroke_data: dict, min_dt_ms: float = 20.0) -> list[np.ndarray]:
    """ストロークデータから各ストロークの速度列を抽出する。

    高サンプリングレートのデータ（dt=0 の連続点が多い）に対して安定した
    速度を算出するため、以下の処理を行う:
    1) 座標に移動平均スムージング（window=5）を適用
    2) スムージング後の座標で累積距離・累積時間を計算し、
       累積時間が min_dt_ms 以上になった時点で速度を確定

    速度 = 累積距離 / (累積時間 / 1000.0) （正規化座標/秒）
    """
    sequences = []
    for stroke in stroke_data.get("strokes", []):
        if len(stroke) < 2:
            continue
        arr = np.array(stroke, dtype=float)
        xs, ys, ts = arr[:, 0], arr[:, 1], arr[:, 3]

        # 座標をスムージング（ノイズの増幅を抑制）
        if len(xs) > 5:
            kernel = np.ones(5) / 5
            xs = np.convolve(xs, kernel, mode="same")
            ys = np.convolve(ys, kernel, mode="same")

        # 累積距離・累積時間で安定した速度を算出
        speeds = []
        accum_dist = 0.0
        accum_t = 0.0
        for i in range(1, len(arr)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            accum_dist += np.sqrt(dx**2 + dy**2)
            accum_t += ts[i] - ts[i - 1]
            if accum_t >= min_dt_ms:
                speeds.append(accum_dist / (accum_t / 1000.0))
                accum_dist = 0.0
                accum_t = 0.0
        if speeds:
            sequences.append(np.array(speeds).reshape(-1, 1))
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
        # パス長で正規化（ポイント数に依存しないようにする）
        normalized_dist = alignment.distance / len(alignment.index1)
        distances.append(normalized_dist)

    # ストローク数の差にペナルティ
    extra = abs(len(ref_seqs) - len(user_seqs))
    if extra > 0:
        penalty = max(distances) if distances and max(distances) > 0 else 1.0
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
        normalized_dist = alignment.distance / len(alignment.index1)
        distances.append(normalized_dist)

    extra = abs(len(ref_seqs) - len(user_seqs))
    if extra > 0:
        penalty = max(distances) if distances and max(distances) > 0 else 1.0
        distances.extend([penalty] * extra)

    mean_dist = np.mean(distances)
    return 1.0 / (1.0 + mean_dist)


def compute_dtw_speed_similarity(ref_data: dict, user_data: dict) -> float:
    """ストロークごとの速度列の DTW 距離の平均から類似度を算出する。

    類似度 = 1 / (1 + mean_distance)

    Returns:
        (0, 1] の範囲の類似度スコア
    """
    ref_seqs = _extract_speed_sequences(ref_data)
    user_seqs = _extract_speed_sequences(user_data)

    if not ref_seqs or not user_seqs:
        return 0.0

    distances = []
    n_pairs = min(len(ref_seqs), len(user_seqs))

    for i in range(n_pairs):
        alignment = dtw(ref_seqs[i], user_seqs[i])
        normalized_dist = alignment.distance / len(alignment.index1)
        distances.append(normalized_dist)

    extra = abs(len(ref_seqs) - len(user_seqs))
    if extra > 0:
        penalty = max(distances) if distances and max(distances) > 0 else 1.0
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


def compute_three_scores(ref_data: dict, user_data: dict) -> dict:
    """見た目スコア・身体的スコア・複合スコアの3種類を算出する。

    - 見た目スコア: SSIM のみ
    - 身体的スコア: 0.2×DTW(xy) + 0.4×DTW(筆圧) + 0.4×DTW(速度)
    - 複合スコア: 0.5×見た目 + 0.5×身体的

    Returns:
        {
            "dtw_xy_similarity": float,
            "dtw_pressure_similarity": float,
            "dtw_speed_similarity": float,
            "ssim_similarity": float,
            "visual_score": float,
            "physical_score": float,
            "composite_score": float,
        }
    """
    dtw_xy = compute_dtw_similarity(ref_data, user_data)
    dtw_pressure = compute_dtw_pressure_similarity(ref_data, user_data)
    dtw_speed = compute_dtw_speed_similarity(ref_data, user_data)
    ssim_sim = compute_ssim_similarity(ref_data, user_data)

    visual_score = ssim_sim
    physical_score = 0.2 * dtw_xy + 0.4 * dtw_pressure + 0.4 * dtw_speed
    composite_score = 0.5 * visual_score + 0.5 * physical_score

    return {
        "dtw_xy_similarity": round(dtw_xy, 6),
        "dtw_pressure_similarity": round(dtw_pressure, 6),
        "dtw_speed_similarity": round(dtw_speed, 6),
        "ssim_similarity": round(ssim_sim, 6),
        "visual_score": round(visual_score, 6),
        "physical_score": round(physical_score, 6),
        "composite_score": round(composite_score, 6),
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
        scores = compute_three_scores(ref_map[character], user_data)
        print(
            f"Visual={scores['visual_score']:.4f} "
            f"Physical={scores['physical_score']:.4f} "
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
        "dtw_speed_similarity",
        "ssim_similarity",
        "visual_score",
        "physical_score",
        "composite_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n結果を保存しました: {output_path} ({len(results)} 件)")


if __name__ == "__main__":
    main()
