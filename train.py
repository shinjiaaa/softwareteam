"""
model train
"""
import os
from ultralytics import YOLO
import wandb
import weave
from openai import OpenAI
import multiprocessing

os.environ['WANDB_API_KEY'] = '1be9ca6360255fbdf54e79e544919c5a40f662b2'
weave.init('tlswldk122104-gnu/intro-example')
client = OpenAI()

# 2. AI 리포트 생성
@weave.op()
def create_completion(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are an expert data scientist that summarizes YOLO training results clearly."},
            {"role": "user", "content": message}
        ],
    )
    return response.choices[0].message.content

def train_yolo():
    data_yaml = r"C:\Users\lab\softwareteam\datasets\ddos_dataset\ddos_dataset.yaml"
    project_name = "ddos_yolov8n"

    # YOLOv8 nano 모델 로드
    model = YOLO("yolov8n.pt")

    # 학습
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=5,
        name=project_name,
        project="runs/detect",
        device=0,
        workers=4
    )

    # model 평가
    metrics = model.val()
    f1_score = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

    print("\n=== 평가 지표 ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Precision, Recall 배열 전체 출력
    precision = metrics.box.p
    recall = metrics.box.r

    # 전체 배열 출력
    print("Precision (per class):", np.round(precision, 4))
    print("Recall    (per class):", np.round(recall, 4))

    # 평균값 출력
    print(f"Precision (mean): {np.mean(precision):.4f}")
    print(f"Recall    (mean): {np.mean(recall):.4f}")

    # F1-score는 이미 float라면 그대로 출력
    print(f"F1-score: {f1_score:.4f}")

    # W&B 로깅
    wandb.init(project="YOLO-Detection", name=project_name)
    wandb.log({
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision": metrics.box.p,
        "Recall": metrics.box.r,
        "F1-score": f1_score
    })
    wandb.finish()

    # AI 리포트
    summary_message = (
        f"The YOLOv8 detection model has been trained.\n"
        f"Results:\n"
        f"- mAP50: {metrics.box.map50:.4f}\n"
        f"- mAP50-95: {metrics.box.map:.4f}\n"
        f"- Precision: {metrics.box.p:.4f}\n"
        f"- Recall: {metrics.box.r:.4f}\n"
        f"- F1-score: {f1_score:.4f}\n"
        f"Generate a concise natural-language analysis of model performance."
    )
    print("\n=== AI 리포트 ===")
    print(create_completion(summary_message))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_yolo()
