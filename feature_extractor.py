"""
特徴量抽出モジュール

筆記データから身体的特徴量を抽出する。
入力: stroke_data = {"strokes": [[[x, y, pressure, time], ...], ...]}
"""

import numpy as np


def extract_features(stroke_data: dict) -> dict:
    """筆記データから特徴量を抽出する。

    Args:
        stroke_data: {"strokes": [[[x, y, pressure, time], ...], ...], ...}

    Returns:
        特徴量の辞書
    """
    strokes = stroke_data["strokes"]

    if not strokes:
        raise ValueError("ストロークデータが空です")

    # 全ストロークを結合して全体の特徴量を計算
    all_points = []
    for stroke in strokes:
        all_points.extend(stroke)
    all_points = np.array(all_points, dtype=float)

    pressures = all_points[:, 2]
    times = all_points[:, 3]

    # --- 筆圧特徴量 ---
    pressure_features = _extract_phase_features(pressures, "pressure")

    # --- 速度特徴量 ---
    speeds = _compute_speeds(strokes)
    speed_features = _extract_phase_features(speeds, "speed")

    # --- 加速度特徴量 ---
    accelerations = _compute_accelerations(strokes)
    accel_features = {
        "acceleration_mean": (
            float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0
        ),
        "acceleration_max_positive": (
            float(np.max(accelerations)) if len(accelerations) > 0 else 0.0
        ),
        "acceleration_max_negative": (
            float(np.min(accelerations)) if len(accelerations) > 0 else 0.0
        ),
    }

    # --- 時間特徴量 ---
    total_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    pause_count = _count_pauses(strokes)
    time_features = {
        "time_total_duration": total_duration,
        "time_pause_count": pause_count,
    }

    # --- 滑らかさ特徴量 ---
    jerk = _compute_jerk(strokes)
    smoothness_features = {
        "smoothness_jerk": jerk,
    }

    # 全特徴量をまとめる
    features = {}
    features.update(pressure_features)
    features.update(speed_features)
    features.update(accel_features)
    features.update(time_features)
    features.update(smoothness_features)

    return features


def _extract_phase_features(values: np.ndarray, prefix: str) -> dict:
    """値の配列から全体・フェーズ別の統計量を抽出する。

    フェーズ: start(最初20%), mid(中間60%), end(最後20%)
    """
    n = len(values)
    if n == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_start": 0.0,
            f"{prefix}_mid": 0.0,
            f"{prefix}_end": 0.0,
        }

    start_end = max(1, int(n * 0.2))
    mid_start = start_end
    mid_end = n - start_end

    # mid区間が空になる場合の対処
    if mid_start >= mid_end:
        mid_start = n // 3
        mid_end = 2 * n // 3
        if mid_start >= mid_end:
            mid_start = 0
            mid_end = n

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_start": float(np.mean(values[:start_end])),
        f"{prefix}_mid": float(np.mean(values[mid_start:mid_end])),
        f"{prefix}_end": float(np.mean(values[-start_end:])),
    }


def _compute_speeds(strokes: list) -> np.ndarray:
    """各ストロークの連続する点間の速度を計算する。"""
    all_speeds = []

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        arr = np.array(stroke, dtype=float)
        xs, ys, _, ts = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        for i in range(1, len(arr)):
            dt = ts[i] - ts[i - 1]
            if dt <= 0:
                continue
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            dist = np.sqrt(dx**2 + dy**2)
            speed = dist / (dt / 1000.0)  # 正規化座標/秒
            all_speeds.append(speed)

    return np.array(all_speeds) if all_speeds else np.array([0.0])


def _compute_accelerations(strokes: list) -> np.ndarray:
    """各ストロークの加速度を計算する。"""
    all_accels = []

    for stroke in strokes:
        if len(stroke) < 3:
            continue
        arr = np.array(stroke, dtype=float)
        xs, ys, _, ts = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        speeds = []
        speed_times = []
        for i in range(1, len(arr)):
            dt = ts[i] - ts[i - 1]
            if dt <= 0:
                continue
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            dist = np.sqrt(dx**2 + dy**2)
            speeds.append(dist / (dt / 1000.0))
            speed_times.append((ts[i] + ts[i - 1]) / 2.0)

        for i in range(1, len(speeds)):
            dt = speed_times[i] - speed_times[i - 1]
            if dt <= 0:
                continue
            accel = (speeds[i] - speeds[i - 1]) / (dt / 1000.0)
            all_accels.append(accel)

    return np.array(all_accels) if all_accels else np.array([0.0])


def extract_per_stroke_features(stroke_data: dict) -> list[dict]:
    """ストロークごとの特徴量を抽出する。

    Args:
        stroke_data: {"strokes": [[[x, y, pressure, time], ...], ...], ...}

    Returns:
        [
            {
                "stroke_index": 1,
                "pressure_mean": float,
                "pressure_start": float,  # 最初20%
                "pressure_mid": float,    # 中間60%
                "pressure_end": float,    # 最後20%
                "speed_mean": float,
                "speed_start": float,
                "speed_mid": float,
                "speed_end": float,
                "acceleration_mean": float | None,  # 点数不足時はNone
                "smoothness_jerk": float | None,     # 点数不足時はNone
                "duration_ms": float,
            },
            ...
        ]
    """
    strokes = stroke_data["strokes"]
    results = []

    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue

        arr = np.array(stroke, dtype=float)
        pressures = arr[:, 2]
        times = arr[:, 3]

        # 筆圧のフェーズ別特徴量
        p_features = _extract_phase_features(pressures, "pressure")

        # 速度のフェーズ別特徴量（1ストローク分が入ったリストを渡す）
        speeds = _compute_speeds([stroke])
        s_features = _extract_phase_features(speeds, "speed")

        # 加速度（3点以上必要）
        accels = _compute_accelerations([stroke])
        accel_mean = float(np.mean(accels)) if len(stroke) >= 3 else None

        # 躍度・滑らかさ（4点以上必要）
        jerk = _compute_jerk([stroke]) if len(stroke) >= 4 else None

        # 持続時間
        duration_ms = float(times[-1] - times[0])

        results.append(
            {
                "stroke_index": i + 1,
                "pressure_mean": p_features["pressure_mean"],
                "pressure_start": p_features["pressure_start"],
                "pressure_mid": p_features["pressure_mid"],
                "pressure_end": p_features["pressure_end"],
                "speed_mean": s_features["speed_mean"],
                "speed_start": s_features["speed_start"],
                "speed_mid": s_features["speed_mid"],
                "speed_end": s_features["speed_end"],
                "acceleration_mean": accel_mean,
                "smoothness_jerk": jerk,
                "duration_ms": duration_ms,
            }
        )

    return results


def _count_pauses(strokes: list, threshold_ms: float = 100.0) -> int:
    """ストローク間の停止回数を数える。

    threshold_ms以上の時間差がストローク間にある場合を停止とカウント。
    """
    if len(strokes) < 2:
        return 0

    pause_count = 0
    for i in range(1, len(strokes)):
        prev_end_time = strokes[i - 1][-1][3]
        curr_start_time = strokes[i][0][3]
        if curr_start_time - prev_end_time >= threshold_ms:
            pause_count += 1

    return pause_count


def _compute_jerk(strokes: list) -> float:
    """躍度（jerk）ベースの滑らかさ指標を計算する。

    値が小さいほど滑らかな動きを示す。
    """
    all_jerks = []

    for stroke in strokes:
        if len(stroke) < 4:
            continue
        arr = np.array(stroke, dtype=float)
        xs, ys, _, ts = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        # 速度を計算
        speeds = []
        speed_times = []
        for i in range(1, len(arr)):
            dt = ts[i] - ts[i - 1]
            if dt <= 0:
                continue
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            dist = np.sqrt(dx**2 + dy**2)
            speeds.append(dist / (dt / 1000.0))
            speed_times.append((ts[i] + ts[i - 1]) / 2.0)

        if len(speeds) < 3:
            continue

        # 加速度を計算
        accels = []
        accel_times = []
        for i in range(1, len(speeds)):
            dt = speed_times[i] - speed_times[i - 1]
            if dt <= 0:
                continue
            accels.append((speeds[i] - speeds[i - 1]) / (dt / 1000.0))
            accel_times.append((speed_times[i] + speed_times[i - 1]) / 2.0)

        if len(accels) < 2:
            continue

        # 躍度を計算
        for i in range(1, len(accels)):
            dt = accel_times[i] - accel_times[i - 1]
            if dt <= 0:
                continue
            jerk = abs(accels[i] - accels[i - 1]) / (dt / 1000.0)
            all_jerks.append(jerk)

    if all_jerks:
        return float(np.mean(all_jerks))
    return 0.0
