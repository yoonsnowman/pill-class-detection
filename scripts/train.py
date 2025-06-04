# train.py
from ultralytics import YOLO
import os
import argparse
import torch # torch 임포트 추가
import configs.config_paths as cc # 설정 파일 임포트 (cc 별칭 사용)

# --- 장치(Device) 설정 ---
# CUDA 사용 가능 여부에 따라 'cuda' 또는 'cpu'로 자동 설정됩니다.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 전역 설정 변수 (configs/config_paths.py 파일에서 로드) ---
# 학습에 사용할 기본 모델 경로 (사전 학습된 모델)
BASE_MODEL_PATH = cc.WEIGHTS_DIR
# 데이터셋 정보가 담긴 YAML 파일 경로
DATA_YAML_PATH = cc.YAML_DIR
# 학습 결과가 저장될 최상위 폴더
BASE_OUTPUT_DIR = cc.OUTPUT_DIR
# (DEVICE는 위에서 직접 선언)

def train_yolo_model(run_folder_name, batch_size, epochs):
    """
    YOLO 모델 학습을 수행하는 함수입니다.
    지정된 실행 폴더 이름, 배치 크기, 에포크를 사용합니다.
    """
    # 학습 시 적용할 하이퍼파라미터 (필요시 추가/수정)
    hyp_aug = dict(
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, degrees=2.0, translate=0.15,
        scale=0.6, shear=0.05, perspective=0.0002, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.1, optimizer='AdamW',
        lr0=0.001, cos_lr=True, patience=50, weight_decay=0.0005
    )

    print("\n✅ YOLO 모델 학습 시작")
    print(f"   - 기본 모델(Pretrained): {BASE_MODEL_PATH}")
    print(f"   - 데이터 YAML: {DATA_YAML_PATH}")
    print(f"   - 출력 최상위 폴더: {BASE_OUTPUT_DIR}")
    print(f"   - 현재 실행 폴더명: {run_folder_name}")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 에포크 수: {epochs}")
    print(f"   - 사용 장치: {DEVICE}") # 직접 선언된 DEVICE 사용

    current_run_path = os.path.join(BASE_OUTPUT_DIR, run_folder_name)
    os.makedirs(current_run_path, exist_ok=True)

    model = YOLO(BASE_MODEL_PATH)
    model.train(
        data=DATA_YAML_PATH,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        project=BASE_OUTPUT_DIR,
        name=run_folder_name,
        device=DEVICE, # 직접 선언된 DEVICE 사용
        exist_ok=True,
        **hyp_aug
    )

    final_model_path_example = os.path.join(current_run_path, 'weights', 'best.pt') # 'best.pt'는 일반적인 경우
    print(f"\n✅ 전체 학습 완료! 결과는 다음 경로에 저장됩니다: {current_run_path}")
    print(f"   주요 모델 가중치 파일 (예상): {final_model_path_example} (또는 'last.pt' 등)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 모델 학습 스크립트. 실행 폴더 이름을 필수로 지정해야 합니다.")
    parser.add_argument('--name', type=str, required=True,
                        help="학습 결과를 저장할 폴더 이름 (예: yolov8_run1). 이 이름으로 'outputs' 폴더 내에 하위 폴더가 생성됩니다.")
    parser.add_argument('--b', type=int, default=32,
                        help="학습 시 사용할 배치 크기 (기본값: 32)")
    parser.add_argument('--e', type=int, default=50,
                        help="학습 시 수행할 에포크 수 (기본값: 50)")
    
    args = parser.parse_args()

    print("--- 학습 설정 (configs/config_paths.py 파일 및 터미널 인자 기반) ---")
    print(f"지정된 실행 폴더 이름: {args.name}")
    print(f"배치 크기: {args.b}")
    print(f"에포크: {args.e}")
    print(f"기본 모델 경로: {BASE_MODEL_PATH}")
    print(f"데이터 YAML 경로: {DATA_YAML_PATH}")
    print(f"출력 최상위 폴더: {BASE_OUTPUT_DIR}")
    print(f"사용 장치: {DEVICE}") # 직접 선언된 DEVICE 사용
    print("----------------------------------------------------------------")
    
    train_yolo_model(
        run_folder_name=args.name,
        batch_size=args.b,
        epochs=args.e
    )