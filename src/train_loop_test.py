# src/train_loop_test.py
from autoconfig import DIR

import torch
import os, sys
def dummy_training():
    print("📁 [경로 확인]")
    train_dir = DIR('data/yolo/train_images')
    print("TRAIN 이미지 경로:", train_dir)

    if not os.path.exists(train_dir):
        print("❌ 경로 없음")
        return

    files = os.listdir(train_dir)
    print(f"총 {len(files)}개 파일 발견")

    print("\n🧠 [장치 확인]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 장치: {device}")

    print("\n🔥 [훈련 시작]")
    for epoch in range(1, 4):
        print(f"Epoch {epoch}/3 ▶ 가짜 손실값: {round(1.0 / epoch, 4)}")

    print("\n✅ 훈련 완료")

if __name__ == "__main__":
    dummy_training()
