# lime_explainer.py
import json
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="OPENAI_API_KEY")  # 환경변수로 관리 권장

def summarize_lime_mask(pos_mask: np.ndarray, neg_mask: np.ndarray) -> str:
    """
    LIME 마스크 요약: 가장 큰 영향력을 가진 영역을 분석.
    pos_mask: 충돌 확률을 높이는 영역
    neg_mask: 충돌 확률을 낮추는 영역
    """
    # 간단한 수치 기반 특징 추출
    pos_intensity = float(np.mean(pos_mask) * 100)
    neg_intensity = float(np.mean(neg_mask) * 100 if neg_mask is not None else 0.0)
    dominant = "positive" if pos_intensity > neg_intensity else "negative"

    description = {
        "dominant": dominant,
        "pos_intensity": pos_intensity,
        "neg_intensity": neg_intensity,
        "analysis": "충돌 확률에 가장 크게 기여한 시각적 영역의 평균 강도값입니다."
    }
    return json.dumps(description, ensure_ascii=False, indent=2)


def generate_lime_explanation(pos_mask: np.ndarray, neg_mask: np.ndarray, class_name: str, collision_prob: float):
    """
    LIME 결과를 LLM을 통해 자연어 설명으로 변환하고, JSON으로 반환.
    """
    lime_summary = summarize_lime_mask(pos_mask, neg_mask)
    prompt = f"""
당신은 영상 기반 충돌 탐지 해석 도우미입니다.
아래는 LIME이 산출한 시각적 근거 요약입니다.

LIME 데이터:
{lime_summary}

대상 객체: {class_name}
충돌 확률: {collision_prob:.2f}

위 데이터를 기반으로,
1. 충돌 위험의 이유
2. 어떤 시각적 요소가 충돌 판단에 가장 영향을 미쳤는지
3. 사람이 이해할 수 있는 짧은 자연어 요약
을 JSON 형태로 반환해 주세요. 예시는 다음과 같습니다:

{{
  "reason": "...",
  "visual_focus": "...",
  "summary": "..."
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    text_output = response.choices[0].message.content
    try:
        return json.loads(text_output)
    except:
        # JSON이 잘 안 맞을 경우 fallback
        return {"summary": text_output.strip()}
