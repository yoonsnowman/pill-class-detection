# train.py
from ultralytics import YOLO
import os
import argparse
import torch
import configs.config_paths as cc

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 명령어
# python src/train.py --model yolov8n --name v8n --b 32 --e 50
# python -m src.train --model yolov8n --name v8n --b 16 --e 10

# --- 전역 설정 변수 ---
DATA_YAML_PATH = cc.YAML_DIR
BASE_OUTPUT_DIR = cc.OUTPUT_DIR


def train_yolo_model(model, run_folder_name, batch_size, epochs):
    # 하이퍼파라미터
    hyp_aug = dict(
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=2.0,
        translate=0.15,
        scale=0.6,
        shear=0.05,
        perspective=0.0002,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=50,
        weight_decay=0.0005
    )
    print("\n✅ YOLO 모델 학습 시작")

    current_run_path = os.path.join(BASE_OUTPUT_DIR, run_folder_name)
    os.makedirs(current_run_path, exist_ok=True)

    model.train(
        data=DATA_YAML_PATH,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        project=BASE_OUTPUT_DIR,
        name=run_folder_name,
        device=device,
        exist_ok=True,
        **hyp_aug
    )

    print(f"\n✅ 학습 완료! 저장 경로: {current_run_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 모델 학습 스크립트.")
    parser.add_argument('--model', type=str, required=True,
                        help="모델명 (예: yolov8n)")
    parser.add_argument('--name', type=str, required=True,
                        help="저장 폴더 이름 (예: 8n_32_50)")
    parser.add_argument('--b', type=int, required=True,
                        help="배치 크기")
    parser.add_argument('--e', type=int, required=True,
                        help="에포크 수")

    args = parser.parse_args()

    # 모델 파일 경로
    base_model_actual_path = f"models/{args.model}.pt"

    # 모델 로딩
    if os.path.exists(base_model_actual_path):
        model = YOLO(base_model_actual_path)
    else:
        model = YOLO(args.model)

    # 모델 학습 실행
    train_yolo_model(
        model=model,
        run_folder_name=args.name,
        batch_size=args.b,
        epochs=args.e
    )
