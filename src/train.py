# train.py
from ultralytics import YOLO
import os
import argparse
import torch
import configs.config_paths as cc
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 명령어: python train.py --model yolov8n --name v8n_32_50 --b 32 --e 50

# --- 전역 설정 변수
# 데이터셋 정보가 담긴 YAML 파일 경로
DATA_YAML_PATH = cc.YAML_DIR
# 학습 결과가 저장될 최상위 폴더
BASE_OUTPUT_DIR = cc.OUTPUT_DIR
# BASE_MODEL_PATH는 이제 터미널 인자로부터 동적으로 생성됩니다.


def train_yolo_model(base_model_actual_path, run_folder_name, batch_size, epochs):
    """
    YOLO 모델 학습을 수행하는 함수입니다.
    동적으로 결정된 기본 모델 경로, 실행 폴더 이름, 배치 크기, 에포크를 사용합니다.
    """
    # 학습 시 적용할 하이퍼파라미터 (필요시 추가/수정)
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
    print(f"   - 기본 모델(Pretrained): {base_model_actual_path}") # 동적으로 생성된 경로 사용
    print(f"   - 데이터 YAML: {DATA_YAML_PATH}")
    print(f"   - 출력 최상위 폴더: {BASE_OUTPUT_DIR}")
    print(f"   - 현재 실행 폴더명: {run_folder_name}")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 에포크 수: {epochs}")
    print(f"   - 사용 장치: {device}")

    current_run_path = os.path.join(BASE_OUTPUT_DIR, run_folder_name)
    os.makedirs(current_run_path, exist_ok=True)

    # 사전 학습된 모델 로드
    model = YOLO(base_model_actual_path)
    # 모델 학습 실행
    model.train(
        data=DATA_YAML_PATH,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        project=BASE_OUTPUT_DIR, # 결과가 저장될 프로젝트 폴더
        name=run_folder_name,    # 실행 이름 (프로젝트 폴더 내에 이 이름으로 하위 폴더 생성)
        device=device,
        exist_ok=True, # 동일한 이름의 실행 폴더가 이미 있을 경우 덮어쓰거나 이어서 학습하도록 허용
        **hyp_aug      # 위에서 정의한 하이퍼파라미터 전달
    )

    # 학습 완료 후, 일반적으로 'best.pt' 모델이 저장됩니다.
    final_model_path_example = os.path.join(current_run_path, 'weights', cc.DEFAULT_MODEL_FILENAME)
    print(f"\n✅ 전체 학습 완료! 결과는 다음 경로에 저장됩니다: {current_run_path}")
    print(f"   주요 모델 가중치 파일 (예상): {final_model_path_example} (또는 'last.pt' 등)")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 모델 학습 스크립트.")
    parser.add_argument('--model', type=str, required=True,
                        help="학습에 사용할 기본 모델의 축약 이름 (예: yolov8n, yolov8s 등). 'models/' 폴더 하위에 해당 .pt 파일이 있어야 합니다.")
    parser.add_argument('--name', type=str, required=True,
                        help="학습 결과를 저장할 폴더 이름 (예: 8n_32_50). 이 이름으로 'outputs' 폴더 내에 하위 폴더가 생성됩니다.")
    parser.add_argument('--b', type=int, required=True, # default 제거, required=True 추가
                        help="학습 시 사용할 배치 크기 (필수 입력)")
    parser.add_argument('--e', type=int, required=True, # default 제거, required=True 추가
                        help="학습 시 수행할 에포크 수 (필수 입력)")
    
    args = parser.parse_args()

    # 터미널에서 받은 모델 축약 이름으로 실제 모델 파일 경로 구성
    # 예: --model yolov8n  =>  models/yolov8n.pt
    base_model_actual_path = f"models/{args.model}.pt"

    if not os.path.exists(base_model_actual_path):
        print(f"⚠️ 지정된 기본 모델 파일을 찾을 수 없습니다: {base_model_actual_path}")
        print(f"   'models/' 폴더 내에 '{args.model}.pt' 파일이 있는지 확인해주세요.")
        exit() # 파일 없으면 종료

    print("--- 학습 설정 (configs/config_paths.py 파일 및 터미널 인자 기반) ---")
    print(f"지정된 실행 폴더 이름: {args.name}")
    print(f"사용할 기본 모델: {args.model} (경로: {base_model_actual_path})")
    print(f"배치 크기: {args.b}")
    print(f"에포크: {args.e}")
    print(f"데이터 YAML 경로: {DATA_YAML_PATH}")
    print(f"출력 최상위 폴더: {BASE_OUTPUT_DIR}")
    print(f"사용 장치: {device}")
    print("----------------------------------------------------------------")
    
    # 모델 학습 함수 호출
    train_yolo_model(
        base_model_actual_path=base_model_actual_path,
        run_folder_name=args.name,
        batch_size=args.b,
        epochs=args.e
    )