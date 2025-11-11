import os, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
        "analysis": "충돌 확률에 가장 크게 기여한 시각적 영역의 평균 강도값입니다.",
    }
    return json.dumps(description, ensure_ascii=False, indent=2)


def generate_lime_explanation(
    pos_mask: np.ndarray, neg_mask: np.ndarray, class_name: str, collision_prob: float
):
    """
    LIME 결과를 LLM을 통해 자연어 설명으로 변환하고, JSON으로 반환.
    사람이 이해하기 쉬운 방식으로 'positive/negative 영역'과 '강도' 개념을 풀어서 설명.
    """
    lime_summary = summarize_lime_mask(pos_mask, neg_mask)
    prompt = f"""
당신은 영상 기반 충돌 탐지 해석 도우미입니다.
아래는 LIME이 산출한 시각적 근거 요약입니다.

LIME 데이터:
{lime_summary}

대상 객체: {class_name}
충돌 확률: {collision_prob:.2f}

**설명 목적:**
LIME은 화면 내에서 충돌 확률을 높이거나 낮추는 영역을 분석합니다.
- positive 영역: 충돌 가능성을 높이는 시각적 단서 (예: 차량의 전면부, 사람의 몸통 등)
- negative 영역: 충돌 가능성을 낮추는 시각적 단서 (예: 배경, 도로, 하늘 등)
- 강도(Intensity): 각 영역이 충돌 판단에 미친 영향의 크기 (값이 높을수록 영향이 큼)

**요청사항:**
1. 충돌 위험이 높다고 판단한 이유를 사람이 납득할 수 있도록 서술하세요.  
2. 'positive'와 'negative' 영역을 실제 시각적 요소로 해석하세요.  
   예: "자동차의 앞부분", "사람의 다리", "도로 가장자리", "장애물 근처" 등.  
3. 강도값을 단순 숫자가 아니라 ‘영향이 매우 강하다 / 약하다’ 식의 자연어로 설명하세요.  
4. 마지막으로 사람이 쉽게 이해할 수 있는 짧은 요약을 포함하세요.  
모든 출력을 JSON 형태로 반환하세요.

예시 출력 형식:
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
        return {"summary": text_output.strip()}
