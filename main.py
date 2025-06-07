import argparse
import subprocess
import sys


def run_preprocess():
    print("\n🚀 [1/5] preprocess.py 실행")
    print("[INFO] YOLO 포맷으로 파싱을 진행합니다.")
    subprocess.run([sys.executable, "-m", "src.preprocess"], check=True)


def run_oversample(list_threshold: int, aug: str, target: int):
    print("\n🚀 [2/5] oversample.py 실행")
    print("[INFO] 소수 클래스 데이터를 증강합니다.")
    subprocess.run([
        sys.executable, "-m", "scripts.oversample",
        "--train_copy",
        "--list", str(list_threshold),
        "--aug", aug,
        "--target", str(target)
    ], check=True)


def run_class_counter():
    print("\n🚀 [3/5] class_counter.py 실행")
    subprocess.run([
        sys.executable, "-m", "scripts.class_counter"
    ], check=True)


def run_train(model: str, name: str, b: int, e: int):
    print(f"\n🚀 [4/5] train.py 실행")
    print("[INFO] 기본값 실행 [MODEL: yolov8x | BATCH: 32 | EPOCHS: 100]")
    print("[INFO] 세부설정 변경 시 train.py 개별 실행이 필요합니다.")
    subprocess.run([
        sys.executable, "-m", "src.train",
        "--model", model,
        "--name", name,
        "--b", str(b),
        "--e", str(e)
    ], check=True)


def run_predict(name: str):
    print(f"\n🚀 [5/5] predict.py 실행")
    print("[INFO] 기본값 실행 [CONF = 0.001 | IOU = 0.45]")
    print("[INFO] 세부설정 변경 시 predict.py 개별 실행이 필요합니다.")
    subprocess.run([
        sys.executable, "-m", "src.predict",
        "--name", name
    ], check=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("⚙️ main.py 실행 감지 [기본값 자동 실행]")
        sys.argv += [
            "--model", "yolov8x",
            "--name", "v8x",
            "--b", "32",
            "--e", "100",
            "--train_copy",
            "--list", "30",
            "--aug", "all",
            "--target", "30"
        ]

    parser = argparse.ArgumentParser(
        description="YOLO 전체 파이프라인 실행 (preprocess → oversample → class_counter2 → train → predict)"
    )
    parser.add_argument("--model", type=str, required=True, help="모델 이름 (예: yolov8n)")
    parser.add_argument("--name", type=str, required=True, help="결과 저장 이름 (예: v8n)")
    parser.add_argument("--b", type=int, required=True, help="배치 크기 (예: 32)")
    parser.add_argument("--e", type=int, required=True, help="에포크 수 (예: 50)")
    parser.add_argument("--train_copy", action="store_true", help="원본 데이터를 복사하여 oversample에 사용")
    parser.add_argument("--list", type=int, default=30, help="오버샘플링 list 기준값")
    parser.add_argument("--aug", type=str, default="all", help="오버샘플링 대상 클래스 (기본: all)")
    parser.add_argument("--target", type=int, default=30, help="오버샘플링 target 수량")

    args = parser.parse_args()

    run_preprocess()
    run_oversample(args.list, args.aug, args.target)
    run_class_counter()
    run_train(args.model, args.name, args.b, args.e)
    run_predict(args.name)
