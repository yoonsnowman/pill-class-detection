# test_config.py
import os, sys
from config import DIR



##############
import torch

def main():
    print("📁 [경로 확인]")
    for key, path in DIR.items():
        print(f"{key:15}: {path}")

    print("\n🧠 [GPU 확인]")
    if torch.cuda.is_available():
        print(f"CUDA 사용 가능! ▶ 현재 장치: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA 사용 불가 (CPU만 사용 중)")

    print("\n📷 [train/test 이미지 파일 개수 확인]")

    # 🔹 train 이미지 수
    train_path = DIR['yolo_train']
    if os.path.exists(train_path):
        train_imgs = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
        print(f"✔ TRAIN 이미지: {len(train_imgs)}개")
        print("예시 파일:", train_imgs[:3])
    else:
        print("❌ TRAIN 경로가 존재하지 않음")

    # 🔹 test 이미지 수
    test_path = DIR['yolo_test']
    if os.path.exists(test_path):
        test_imgs = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
        print(f"✔ TEST 이미지: {len(test_imgs)}개")
        print("예시 파일:", test_imgs[:3])
    else:
        print("❌ TEST 경로가 존재하지 않음")

if __name__ == "__main__":
    main()
