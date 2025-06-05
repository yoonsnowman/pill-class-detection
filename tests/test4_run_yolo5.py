# src/test4_run_yolo5.py
import os
import subprocess

def run_yolov5_train():
    data_yaml   = 'data/yolo/raw_split/data.yaml'
    project_dir = 'data/yolo/raw_split/output'
    weights     = 'yolov5s.pt'
    name        = 'pill_yolo'

    os.makedirs(project_dir, exist_ok=True)

    # ✅ 안전한 subprocess 명령어 리스트
    command = [
        'python', 'models/yolov5/train.py',
        '--img', '640',
        '--batch', '16',
        '--epochs', '20',
        '--data', data_yaml,
        '--weights', weights,
        '--project', project_dir,
        '--name', name
    ]

    print("\n🚀 실행 명령어:")
    print(" ".join(f'"{arg}"' if ' ' in arg else arg for arg in command))
    print()


    subprocess.run(command, check=True)

if __name__ == '__main__':
    run_yolov5_train()
