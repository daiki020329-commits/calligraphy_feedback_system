"""
差分計算モジュール

お手本とユーザーの特徴量の差分を計算し、
LLM用のテキストに変換する。
"""

# 特徴量の日本語ラベルと閾値設定
FEATURE_CONFIG = {
    "pressure_mean": {"label": "筆圧（平均）", "unit": ""},
    "pressure_std": {"label": "筆圧（ばらつき）", "unit": ""},
    "pressure_start": {"label": "筆圧（入り）", "unit": ""},
    "pressure_mid": {"label": "筆圧（中盤）", "unit": ""},
    "pressure_end": {"label": "筆圧（抜き）", "unit": ""},
    "speed_mean": {"label": "速度（平均）", "unit": "/秒"},
    "speed_std": {"label": "速度（ばらつき）", "unit": "/秒"},
    "speed_start": {"label": "速度（書き出し）", "unit": "/秒"},
    "speed_mid": {"label": "速度（中盤）", "unit": "/秒"},
    "speed_end": {"label": "速度（書き終わり）", "unit": "/秒"},
    "acceleration_mean": {"label": "加速度（平均）", "unit": ""},
    "acceleration_max_positive": {"label": "加速度（最大加速）", "unit": ""},
    "acceleration_max_negative": {"label": "加速度（最大減速）", "unit": ""},
    "time_total_duration": {"label": "総書字時間", "unit": "ms"},
    "time_pause_count": {"label": "停止回数", "unit": "回"},
    "smoothness_jerk": {"label": "滑らかさ（躍度）", "unit": ""},
}


def compute_diff(ref_features: dict, user_features: dict) -> dict:
    """お手本とユーザーの特徴量の差分を計算する。

    Args:
        ref_features: お手本の特徴量
        user_features: ユーザーの特徴量

    Returns:
        差分情報の辞書
    """
    diffs = {}

    for key in ref_features:
        if key not in user_features:
            continue

        ref_val = ref_features[key]
        user_val = user_features[key]
        abs_diff = user_val - ref_val

        # パーセンテージ差分
        if abs(ref_val) > 1e-8:
            pct_diff = (abs_diff / abs(ref_val)) * 100.0
        else:
            pct_diff = 0.0 if abs(abs_diff) < 1e-8 else float("inf")

        diffs[key] = {
            "ref": ref_val,
            "user": user_val,
            "abs_diff": abs_diff,
            "pct_diff": pct_diff,
        }

    return diffs


def diff_to_text(diffs: dict, threshold_pct: float = 10.0) -> str:
    """差分情報をLLM用テキストに変換する。

    閾値以上の差がある特徴量のみテキスト化する。

    Args:
        diffs: compute_diffの出力
        threshold_pct: テキスト化する最小パーセンテージ差

    Returns:
        LLMに渡すテキスト
    """
    lines = []
    lines.append("【身体的データの比較結果】\n")

    significant_diffs = []
    minor_diffs = []

    for key, diff_info in diffs.items():
        config = FEATURE_CONFIG.get(key)
        if config is None:
            continue

        label = config["label"]
        ref_val = diff_info["ref"]
        user_val = diff_info["user"]
        pct = diff_info["pct_diff"]

        # 差分の方向を判定
        direction = _get_direction(key, diff_info["abs_diff"], pct)

        if abs(pct) >= threshold_pct:
            line = f"- {label}: お手本 {ref_val:.3f} → ユーザー {user_val:.3f}（{pct:+.0f}%、{direction}）"
            significant_diffs.append((abs(pct), line))
        else:
            minor_diffs.append(label)

    if significant_diffs:
        # 差の大きい順にソート
        significant_diffs.sort(key=lambda x: x[0], reverse=True)
        lines.append("■ 顕著な差がある項目:")
        for _, line in significant_diffs:
            lines.append(line)
    else:
        lines.append("■ 全体的にお手本に近い書き方です。")

    if minor_diffs:
        lines.append(f"\n■ お手本と近い項目: {', '.join(minor_diffs)}")

    return "\n".join(lines)


def _get_direction(key: str, abs_diff: float, pct: float) -> str:
    """差分の方向を自然な日本語で表現する。"""
    if abs(pct) < 10:
        return "ほぼ同じ"

    is_higher = abs_diff > 0

    direction_map = {  # 「高い」「低い」だけでなく、特徴量ごとに自然な表現にする
        "pressure": ("強すぎる", "弱すぎる"),
        "speed": ("速すぎる", "遅すぎる"),
        "acceleration": ("急すぎる", "緩やかすぎる"),
        "time_total_duration": ("時間がかかりすぎ", "急ぎすぎ"),
        "time_pause_count": ("停止が多い", "停止が少ない"),
        "smoothness_jerk": ("ぎこちない", "滑らかすぎる"),
    }

    for prefix, (high_label, low_label) in direction_map.items():
        if key.startswith(prefix):
            return high_label if is_higher else low_label

    return "高い" if is_higher else "低い"
