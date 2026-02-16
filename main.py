"""
書道フィードバックシステム - メインアプリ

全モジュールを統合したデモアプリケーション。

使い方:
    python main.py --ref <お手本JSON> --user <ユーザーJSON>
    python main.py --demo  （デモデータで実行）
"""

import argparse
import io
import json
import os
import sys

# Windows環境でのUTF-8出力対応
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from feature_extractor import extract_features
from image_generator import render_stroke_to_image, create_comparison_image
from diff_calculator import compute_diff, diff_to_text
from feedback_generator import generate_feedback


def load_stroke_data(filepath: str) -> dict:
    """JSONファイルから筆記データを読み込む。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def create_demo_data() -> tuple[dict, dict]:
    """デモ用のお手本・ユーザーデータを生成する。

    「一」（横棒）を模した簡単なストロークデータ。
    """
    # お手本: 滑らかで安定した横棒
    ref_data = {
        "strokes": [
            [
                [0.2, 0.5, 0.6, 0],
                [0.25, 0.5, 0.7, 30],
                [0.3, 0.5, 0.7, 60],
                [0.35, 0.5, 0.7, 90],
                [0.4, 0.5, 0.7, 120],
                [0.45, 0.5, 0.65, 150],
                [0.5, 0.5, 0.65, 180],
                [0.55, 0.5, 0.6, 210],
                [0.6, 0.5, 0.6, 240],
                [0.65, 0.5, 0.55, 270],
                [0.7, 0.5, 0.5, 300],
                [0.75, 0.5, 0.4, 330],
                [0.8, 0.5, 0.3, 360],
            ]
        ],
        "stroke_count": 1,
        "canvas_size": [600, 600],
        "character": "一",
    }

    # ユーザー: 筆圧が強く、やや波打つ横棒
    user_data = {
        "strokes": [
            [
                [0.18, 0.52, 0.8, 0],
                [0.24, 0.51, 0.85, 40],
                [0.3, 0.49, 0.9, 80],
                [0.36, 0.50, 0.85, 110],
                [0.42, 0.52, 0.85, 150],
                [0.48, 0.51, 0.8, 190],
                [0.54, 0.49, 0.8, 230],
                [0.60, 0.50, 0.75, 270],
                [0.66, 0.52, 0.7, 310],
                [0.72, 0.51, 0.65, 350],
                [0.78, 0.50, 0.6, 400],
                [0.82, 0.51, 0.5, 450],
            ]
        ],
        "stroke_count": 1,
        "canvas_size": [600, 600],
        "character": "一",
    }

    return ref_data, user_data


def run_feedback(ref_data: dict, user_data: dict, output_dir: str = "output"):
    """フィードバック生成パイプラインを実行する。"""
    character = ref_data.get("character", "?")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== 書道フィードバックシステム ===")
    print(f"対象文字: {character}\n")

    # 1. 特徴量抽出
    print("[1/4] 特徴量を抽出中...")
    ref_features = extract_features(ref_data)
    user_features = extract_features(user_data)

    print(f"  お手本 - 筆圧平均: {ref_features['pressure_mean']:.3f}, "
          f"速度平均: {ref_features['speed_mean']:.3f}")
    print(f"  ユーザー - 筆圧平均: {user_features['pressure_mean']:.3f}, "
          f"速度平均: {user_features['speed_mean']:.3f}")

    # 2. 画像生成
    print("\n[2/4] 比較画像を生成中...")
    comparison_image = create_comparison_image(ref_data, user_data)

    # 個別画像も保存
    ref_image = render_stroke_to_image(ref_data)
    user_image = render_stroke_to_image(user_data)

    ref_path = os.path.join(output_dir, f"{character}_ref.png")
    user_path = os.path.join(output_dir, f"{character}_user.png")
    comp_path = os.path.join(output_dir, f"{character}_comparison.png")

    with open(ref_path, "wb") as f:
        f.write(ref_image)
    with open(user_path, "wb") as f:
        f.write(user_image)
    with open(comp_path, "wb") as f:
        f.write(comparison_image)

    print(f"  保存先: {comp_path}")

    # 3. 差分計算
    print("\n[3/4] 差分を計算中...")
    diffs = compute_diff(ref_features, user_features)
    body_data_text = diff_to_text(diffs)
    print(body_data_text)

    # 4. フィードバック生成
    print("\n[4/4] フィードバックを生成中...")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[警告] 環境変数 ANTHROPIC_API_KEY が設定されていません。")
        print("  フィードバック生成をスキップします。")
        print("  設定方法: set ANTHROPIC_API_KEY=your-api-key")

        # 差分テキストだけ保存
        result_path = os.path.join(output_dir, f"{character}_diff.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(body_data_text)
        print(f"\n  差分データを保存しました: {result_path}")
        return

    feedback = generate_feedback(
        comparison_image=comparison_image,
        body_data_text=body_data_text,
        character=character,
    )

    print(f"\n{'='*50}")
    print("【フィードバック】")
    print(f"{'='*50}")
    print(feedback)
    print(f"{'='*50}")

    # 結果を保存
    result_path = os.path.join(output_dir, f"{character}_feedback.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"対象文字: {character}\n\n")
        f.write(body_data_text)
        f.write(f"\n\n{'='*50}\n")
        f.write("【フィードバック】\n")
        f.write(feedback)

    print(f"\n結果を保存しました: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="書道フィードバックシステム")
    parser.add_argument("--ref", type=str, help="お手本データのJSONファイルパス")
    parser.add_argument("--user", type=str, help="ユーザーデータのJSONファイルパス")
    parser.add_argument("--demo", action="store_true", help="デモデータで実行")
    parser.add_argument("--output", type=str, default="output", help="出力ディレクトリ")
    args = parser.parse_args()

    if args.demo:
        print("デモモードで実行します...\n")
        ref_data, user_data = create_demo_data()
        run_feedback(ref_data, user_data, args.output)

    elif args.ref and args.user:
        ref_data = load_stroke_data(args.ref)
        user_data = load_stroke_data(args.user)
        run_feedback(ref_data, user_data, args.output)

    else:
        parser.print_help()
        print("\n使用例:")
        print("  python main.py --demo")
        print("  python main.py --ref calligraphy_data/ref.json --user calligraphy_data/user.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
