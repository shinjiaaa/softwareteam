import os
import numpy as np
import wandb
import weave
from dotenv import load_dotenv
from ultralytics import YOLO
from openai import OpenAI
import multiprocessing

# 환경 변수 로드 (.env - API key 유출 안 되게 해 주세요!!!)
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
weave.init("tlswldk122104-gnu/intro-example")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)


# AI 리포트 함수 (YOLO 성능을 자연어로 요약 - 이건 유료라서... 모델 개선 다 되면 요약 지우고 wandb만 남길게요)
@weave.op()
def create_completion(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are an expert data scientist that summarizes YOLO evaluation results clearly."},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


# 모델 평가 함수
def evaluate_yolo():
    model_path = r"C:\Users\lab\softwareteam\runs\detect\VisDrone_train4\weights\best.pt"
    data_yaml = r"C:\Users\lab\softwareteam\VisDrone.yaml"

    print(f"\n📊 평가 시작: {model_path}")
    model = YOLO(model_path)

    # 모델 평가
    metrics = model.val(data=data_yaml, device=0)
    f1_score = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

    print("\n=== 평가 결과 ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {np.mean(metrics.box.p):.4f}")
    print(f"Recall    (mean): {np.mean(metrics.box.r):.4f}")
    print(f"F1-score (mean): {np.mean(f1_score):.4f}")

    # 클래스별 지표
    print("\n=== 클래스별 Precision / Recall ===")
    print("Precision:", np.round(metrics.box.p, 4))
    print("Recall   :", np.round(metrics.box.r, 4))

    # W&B 로깅
    wandb.init(project="YOLO-Evaluation", name="VisDrone_best_eval")
    wandb.log({
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision_mean": np.mean(metrics.box.p),
        "Recall_mean": np.mean(metrics.box.r),
        "F1_score_mean": np.mean(f1_score)
    })
    wandb.finish()

    # 리포트
    summary_message = (
        f"The YOLO detection model has been evaluated.\n"
        f"Results:\n"
        f"- mAP50: {metrics.box.map50:.4f}\n"
        f"- mAP50-95: {metrics.box.map:.4f}\n"
        f"- Precision(mean): {np.mean(metrics.box.p):.4f}\n"
        f"- Recall(mean): {np.mean(metrics.box.r):.4f}\n"
        f"- F1(mean): {np.mean(f1_score):.4f}\n"
        f"Generate a concise natural-language analysis of model performance."
    )

    print(create_completion(summary_message))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    evaluate_yolo()
