# src/train_loop.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # ← 요거 추가

import torch
from configs.config_sh import DIR

def dummy_training():
    print("📁 [경로 확인]")
    train_dir = DIR['raw_train']
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
