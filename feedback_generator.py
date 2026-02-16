"""
フィードバック生成モジュール

Anthropic API (Claude) を使用して、
書道学習者へのフィードバックを生成する。
"""

import base64
import os

import anthropic


SYSTEM_PROMPT = """あなたは書道の先生です。生徒が書いた文字について、お手本と比較してフィードバックを提供してください。

## 分析の観点
1. 画像から形状を分析してください（バランス、はね、とめ、払い、線の太さの変化）
2. 身体的データから書き方の特徴を分析してください（筆圧のかけ方、運筆の速度）

## フィードバックのルール
- 数値をそのまま伝えない（「40%強い」ではなく「力が入りすぎています」のように表現）
- 比喩やイメージを使って伝える（「羽が落ちるように軽く」「川の流れのように滑らかに」など）
- 具体的な身体の動かし方を提案する（「息を吐きながら」「手首を柔らかく」など）
- 良い点を必ず1つ含める
- 3〜5文で簡潔にまとめる
- 書道の専門用語は最小限にし、初心者にも分かりやすく"""

SYSTEM_PROMPT_MULTITURN = """あなたは書道の先生です。生徒が繰り返し練習しており、毎回お手本と比較したフィードバックを求めています。

## 分析の観点
1. 画像から形状を分析してください（バランス、はね、とめ、払い、線の太さの変化）
2. 身体的データから書き方の特徴を分析してください（筆圧のかけ方、運筆の速度）
3. 前回の試みと比較して、改善された点・まだ課題が残る点を指摘してください

## フィードバックのルール
- 数値をそのまま伝えない（「40%強い」ではなく「力が入りすぎています」のように表現）
- 比喩やイメージを使って伝える（「羽が落ちるように軽く」「川の流れのように滑らかに」など）
- 具体的な身体の動かし方を提案する（「息を吐きながら」「手首を柔らかく」など）
- 良い点を必ず1つ含める
- 3〜5文で簡潔にまとめる
- 書道の専門用語は最小限にし、初心者にも分かりやすく
- 2回目以降は、前回からの変化に必ず言及する（「前回より〜が良くなりました」など）"""

MAX_CONVERSATION_TURNS = 5


def generate_feedback(
    comparison_image: bytes,
    body_data_text: str,
    character: str = "",
    model: str = "claude-sonnet-4-5-20250929",
) -> str:
    """比較画像と身体的データからフィードバックを生成する。

    Args:
        comparison_image: 比較画像のPNGバイト列
        body_data_text: 身体的データの差分テキスト
        character: 対象の文字（例: "永"）
        model: 使用するClaudeモデル

    Returns:
        フィードバックテキスト
    """
    client = anthropic.Anthropic()  # ここでAPIキーを環境変数から受け取る

    image_b64 = base64.standard_b64encode(comparison_image).decode("utf-8")

    user_message = f"以下は「{character}」の書道練習の結果です。\n\n"
    user_message += "左がお手本、右が生徒の作品です。\n\n"
    user_message += body_data_text
    user_message += (
        "\n\n画像と身体的データを総合的に分析して、フィードバックをお願いします。"
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_message,
                    },
                ],
            }
        ],
    )

    return response.content[0].text


def generate_feedback_multiturn(
    comparison_image: bytes,
    body_data_text: str,
    character: str = "",
    conversation_history: list[dict] | None = None,
    attempt_number: int = 1,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[list[dict], str]:
    """マルチターン会話でフィードバックを生成する。

    会話履歴を受け取り、新しいユーザーメッセージを追加してAPI呼び出しを行う。
    画像によるコンテキスト膨張を防ぐため、会話は最大5ターンに制限する。

    Args:
        comparison_image: 比較画像のPNGバイト列
        body_data_text: 身体的データの差分テキスト
        character: 対象の文字（例: "永"）
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
    # 各ターン = user + assistant の2メッセージ
    if len(conversation_history) >= MAX_CONVERSATION_TURNS * 2:
        # 最新の (MAX_CONVERSATION_TURNS - 1) ターン分を残す
        keep = (MAX_CONVERSATION_TURNS - 1) * 2
        conversation_history = conversation_history[-keep:]

    image_b64 = base64.standard_b64encode(comparison_image).decode("utf-8")

    if attempt_number == 1:
        user_text = f"以下は「{character}」の書道練習の結果です。\n\n"
        user_text += "左がお手本、右が生徒の作品です。\n\n"
    else:
        user_text = f"「{character}」の{attempt_number}回目の練習結果です。\n\n"
        user_text += "左がお手本、右が今回の作品です。\n\n"

    user_text += body_data_text
    user_text += "\n\n画像と身体的データを総合的に分析して、フィードバックをお願いします。"

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            },
            {
                "type": "text",
                "text": user_text,
            },
        ],
    }

    messages = conversation_history + [user_message]

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT_MULTITURN,
        messages=messages,
    )

    feedback_text = response.content[0].text

    # 会話履歴を更新
    conversation_history.append(user_message)
    conversation_history.append({
        "role": "assistant",
        "content": feedback_text,
    })

    return conversation_history, feedback_text
