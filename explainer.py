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
    lime_summary = summarize_lime_mask(pos_mask, neg_mask)
    prompt = f"""
당신은 **영상 기반 충돌 탐지 해석 도우미**입니다.
LIME이 시각적으로 분석한 결과를 사람이 이해할 수 있는 방식으로 설명해야 합니다.

### 📘 입력 정보
- LIME 데이터: {lime_summary}
- 대상 객체: {class_name}
- 충돌 확률: {collision_prob:.2f}

### 🎯 목표
사람이 이 시스템의 해석을 읽고 “아, 이래서 위험하다고 판단했구나”라고 납득할 수 있게 설명하세요.

### 🧩 LIME 개념 설명 (참고용)
- **positive 영역**: 충돌 가능성을 높이는 시각적 단서입니다.  
  (예: 차량의 전면부, 사람의 몸통, 화면 중앙의 큰 물체)
- **negative 영역**: 충돌 가능성을 낮추는 시각적 단서입니다.  
  (예: 하늘, 도로, 배경, 안정적인 영역)
- **강도(Intensity)**: 각 영역이 충돌 판단에 미친 영향의 크기입니다.  
  (값이 높을수록 해당 영역이 모델 판단에 큰 영향을 미쳤다는 뜻입니다.)

### 🧠 설명 방식
1. **Reason (충돌 이유)**  
   충돌 확률이 높다고 판단한 이유를 논리적으로 서술합니다.  
   예: “전방의 차량이 급격히 확대되어 접근하고 있기 때문입니다.”

2. **Visual Focus (시각적 초점)**  
   ‘positive/negative’ 영역을 실제 장면 속 물체로 구체적으로 해석합니다.  
   예: “차량의 앞부분”, “사람의 다리”, “도로 중앙선 근처”, “하늘 영역”

3. **Intensity Interpretation (강도 해석)**  
   강도값을 단순 수치 대신 ‘영향이 매우 강함/중간 정도/약함’으로 자연어화합니다.  
   예: “해당 영역은 충돌 판단에 매우 강한 영향을 미쳤습니다.”

4. **Summary (요약)**  
   사람이 읽었을 때 한 문장으로 이해할 수 있도록 자연스럽게 요약합니다.

### 🧾 출력 형식 (JSON)
{{
  "reason": "...",
  "visual_focus": "...",
  "intensity_interpretation": "...",
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
