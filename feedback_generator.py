"""
フィードバック生成モジュール

Anthropic API (Claude) を使用して、
書道学習者へのフィードバックを生成する。
"""

import base64
import os

import anthropic

SYSTEM_PROMPT = """あなたは書道の先生です。生徒が書いた文字について、お手本と比較してフィードバックを提供してください。

## 入力データの説明
- 画像1（左右比較）: 左がお手本、右が生徒の作品です。それぞれの全体像を確認できます
- 画像2（重ね合わせ）: お手本（赤）と生徒の作品（青）を同じ座標上に重ねた画像です。ズレている箇所が視覚的にわかります
- 身体的データ: ストローク（画）ごとの筆圧・速度・加速度・滑らかさが提供されます
  - 「第N画」はN番目に書かれたストロークを意味します
  - 「入り」は書き始め、「中盤」は中間部分、「抜き」は書き終わりに対応します

## 分析の手順
1. まず画像1（左右比較）で、お手本と生徒の作品の形状の違いを把握してください（バランス、はね、とめ、払い、線の太さの変化）
2. 画像2（重ね合わせ）で、具体的にどの部分がどの方向にズレているかを特定してください
3. 対象文字「永」の筆順知識と画像の線の形状から、身体的データの「第N画」が画像上のどの線に該当するかを推定してください
4. 画像で気になる箇所（特に、字の上手さ、綺麗さに繋がる箇所）について、対応する身体的データを参照し、形状の違いが生じた原因（筆圧・速度・加速度・滑らかさ）を特定してください。勿論、身体的データ関係なしの形状の違いも指摘して構いません。
   - 例: 重ね合わせ画像で青い線が赤い線より太い箇所 → 対応するストロークの筆圧データを確認 → 「第3画（左のはらい）の書き出しで力が入りすぎています」

## フィードバックのルール
- 画像上の具体的な箇所と身体的データを結びつけて指摘する（「第N画の○○の部分で〜」）
- 数値をそのまま伝えない（「40%強い」ではなく「力が入りすぎています」のように表現）
- 具体的な身体の動かし方を提案する（「息を吐きながら」「手首を柔らかく」など）
- 良い点を必ず1つ含める
- 5〜6文で簡潔にまとめる
- ユーザは重ね合わせの比較画像を確認することができないので、重ね合わせ画像の分析はあくまでフィードバックの精度を上げるための内部的な分析であることに留意してください
- 書道の専門用語は最小限にし、初心者にも分かりやすく"""

SYSTEM_PROMPT_MULTITURN = """あなたは書道の先生です。生徒が繰り返し練習しており、毎回お手本と比較したフィードバックを求めています。

## 入力データの説明
- 画像1（左右比較）: 左がお手本、右が生徒の作品です。それぞれの全体像を確認できます
- 画像2（重ね合わせ）: お手本（赤）と生徒の作品（青）を同じ座標上に重ねた画像です。ズレている箇所が視覚的にわかります
- 身体的データ: ストローク（画）ごとの筆圧・速度・加速度・滑らかさが提供されます
  - 「第N画」はN番目に書かれたストロークを意味します
  - 「入り」は書き始め、「中盤」は中間部分、「抜き」は書き終わりに対応します

## 分析の手順
1. まず画像1（左右比較）で、お手本と生徒の作品の形状の違いを把握してください（バランス、はね、とめ、払い、線の太さの変化）
2. 画像2（重ね合わせ）で、具体的にどの部分がどの方向にズレているかを特定してください
3. 対象文字「永」の筆順知識と画像の線の形状から、身体的データの「第N画」が画像上のどの線に該当するかを推定してください
4. 画像で気になる箇所（特に、字の上手さ、綺麗さに繋がる箇所）について、対応する身体的データを参照し、形状の違いが生じた原因（筆圧・速度・加速度・滑らかさ）を特定して、フィードバックを提供してください。勿論、身体的データ関係なしの形状の違いも指摘して構いません。
   - 例: 重ね合わせ画像で青い線が赤い線より太い箇所 → 対応するストロークの筆圧データを確認 → 「第3画（左のはらい）の書き出しで力が入りすぎています」
5. 前回の試みと比較して、改善された点・まだ課題が残る点を指摘してください

## フィードバックのルール
- 画像上の具体的な箇所と身体的データを結びつけて指摘する（「第N画の○○の部分で〜」）
- 数値をそのまま伝えない（「40%強い」ではなく「力が入りすぎています」のように表現）
- 具体的な身体の動かし方を提案する（「息を吐きながら」「手首を柔らかく」など）
- 良い点を必ず1つ含める
- 5〜6文で簡潔にまとめる
- ユーザは重ね合わせの比較画像を確認することができないので、重ね合わせ画像の分析はあくまでフィードバックの精度を上げるための内部的な分析であることに留意してください
- 書道の専門用語は最小限にし、初心者にも分かりやすく
- 2回目以降は、前回からの変化に必ず言及する（「前回より〜が良くなりました」など）"""

MAX_CONVERSATION_TURNS = 5


def generate_feedback(
    comparison_image: bytes,
    body_data_text: str,
    character: str = "",
    overlay_image: bytes | None = None,
    model: str = "claude-sonnet-4-5-20250929",
) -> str:
    """比較画像と身体的データからフィードバックを生成する。

    Args:
        comparison_image: 左右比較画像のPNGバイト列
        body_data_text: 身体的データの差分テキスト
        character: 対象の文字（例: "永"）
        overlay_image: 重ね合わせ画像のPNGバイト列（省略可）
        model: 使用するClaudeモデル

    Returns:
        フィードバックテキスト
    """
    client = anthropic.Anthropic()  # ここでAPIキーを環境変数から受け取る

    comparison_b64 = base64.standard_b64encode(comparison_image).decode("utf-8")

    user_message = f"以下は「{character}」の書道練習の結果です。\n\n"
    user_message += body_data_text
    user_message += (
        "\n\n画像と身体的データを総合的に分析して、フィードバックをお願いします。"
    )

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
            "text": user_message,
        }
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    return response.content[0].text


def generate_feedback_multiturn(
    comparison_image: bytes,
    body_data_text: str,
    character: str = "",
    overlay_image: bytes | None = None,
    conversation_history: list[dict] | None = None,
    attempt_number: int = 1,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[list[dict], str]:
    """マルチターン会話でフィードバックを生成する。

    会話履歴を受け取り、新しいユーザーメッセージを追加してAPI呼び出しを行う。
    画像によるコンテキスト膨張を防ぐため、会話は最大5ターンに制限する。

    Args:
        comparison_image: 左右比較画像のPNGバイト列
        body_data_text: 身体的データの差分テキスト
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
    # 各ターン = user + assistant の2メッセージ
    if len(conversation_history) >= MAX_CONVERSATION_TURNS * 2:
        # 最新の (MAX_CONVERSATION_TURNS - 1) ターン分を残す
        keep = (MAX_CONVERSATION_TURNS - 1) * 2
        conversation_history = conversation_history[-keep:]

    comparison_b64 = base64.standard_b64encode(comparison_image).decode("utf-8")

    if attempt_number == 1:
        user_text = f"以下は「{character}」の書道練習の結果です。\n\n"
    else:
        user_text = f"「{character}」の{attempt_number}回目の練習結果です。\n\n"

    user_text += body_data_text
    user_text += (
        "\n\n画像と身体的データを総合的に分析して、フィードバックをお願いします。"
    )

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
        system=SYSTEM_PROMPT_MULTITURN,
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
