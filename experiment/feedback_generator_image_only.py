"""
画像のみフィードバック生成モジュール（条件B用）

Anthropic API (Claude) を使用して、
比較画像のみから書道学習者へのフィードバックを生成する。
身体的データ（筆圧・速度など）は使用しない。
"""

import base64
import os

import anthropic


SYSTEM_PROMPT_IMAGE_ONLY = """あなたは書道の先生です。生徒が繰り返し練習しており、毎回お手本と比較したフィードバックを求めています。

## 入力データの説明
- 画像1（左右比較）: 左がお手本、右が生徒の作品です。それぞれの全体像を確認できます
- 画像2（重ね合わせ）: お手本（赤）と生徒の作品（青）を同じ座標上に重ねた画像です。ズレている箇所が視覚的にわかります

## 分析の手順
1. まず画像1（左右比較）で、お手本と生徒の作品の形状の違いを把握してください（バランス、はね、とめ、払い、線の太さの変化）
2. 画像2（重ね合わせ）で、具体的にどの部分がどの方向にズレているかを特定してください
3. 画像で気になる箇所（特に、字の上手さ、綺麗さに繋がる箇所）について、フィードバックを提供してください。
4. 前回の試みと比較して、改善された点・まだ課題が残る点を指摘してください

## フィードバックのルール
- 具体的な身体の動かし方を提案する（「息を吐きながら」「手首を柔らかく」など）
- 良い点を必ず1つ含める
- 3〜6文で簡潔にまとめる
- ユーザは重ね合わせの比較画像を確認することができないので、重ね合わせ画像の分析はあくまでフィードバックの精度を上げるための内部的な分析であることに留意してください
- 書道の専門用語は最小限にし、初心者にも分かりやすく
- 2回目以降は、前回からの変化に必ず言及する（「前回より〜が良くなりました」など）"""

MAX_CONVERSATION_TURNS = 5


def generate_feedback_image_only(
    comparison_image: bytes,
    character: str = "",
    overlay_image: bytes | None = None,
    conversation_history: list[dict] | None = None,
    attempt_number: int = 1,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[list[dict], str]:
    """比較画像のみからマルチターンでフィードバックを生成する。

    Args:
        comparison_image: 左右比較画像のPNGバイト列
        character: 対象の文字（例: "永"）
        overlay_image: 重ね合わせ画像のPNGバイト列（省略可）
        conversation_history: これまでの会話履歴（Noneなら新規開始）
        attempt_number: 試行番号（1始まり）
        model: 使用するClaudeモデル

    Returns:
        (更新された会話履歴, フィードバックテキスト) のタプル
    """
    client = anthropic.Anthropic()

    if conversation_history is None:
        conversation_history = []

    # 古いターンを切り詰め（画像でコンテキストが膨らむため）
    if len(conversation_history) >= MAX_CONVERSATION_TURNS * 2:
        keep = (MAX_CONVERSATION_TURNS - 1) * 2
        conversation_history = conversation_history[-keep:]

    comparison_b64 = base64.standard_b64encode(comparison_image).decode("utf-8")

    if attempt_number == 1:
        user_text = f"以下は「{character}」の書道練習の結果です。\n\n"
    else:
        user_text = f"「{character}」の{attempt_number}回目の練習結果です。\n\n"

    user_text += "画像を分析して、フィードバックをお願いします。"

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": comparison_b64,
            },
        },
    ]

    if overlay_image is not None:
        overlay_b64 = base64.standard_b64encode(overlay_image).decode("utf-8")
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": overlay_b64,
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": user_text,
        }
    )

    user_message = {
        "role": "user",
        "content": content,
    }

    messages = conversation_history + [user_message]

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT_IMAGE_ONLY,
        messages=messages,
    )

    feedback_text = response.content[0].text

    # 会話履歴を更新
    conversation_history.append(user_message)
    conversation_history.append(
        {
            "role": "assistant",
            "content": feedback_text,
        }
    )

    return conversation_history, feedback_text
