# main.py
import argparse
import subprocess
import sys

# 기본값 사용
# python -m main

# 설정값 사용(모델pt명, outputs/에 저장할 폴더명, 배치, 에포크)
# python -m main all --model yolov8n --name v8n --b 32 --e 10


def run_preprocess():
    print("\n🚀 [1/3] preprocess.py 실행")
    subprocess.run([
        sys.executable, "-m", "src.preprocess"
    ], check=True)


def run_train(args):
    print(f"\n🚀 [2/3] train.py 실행")
    subprocess.run([
        sys.executable, "-m", "src.train",
        "--model", args.model,
        "--name", args.name,
        "--b", str(args.b),
        "--e", str(args.e)
    ], check=True)


def run_predict(args):
    print(f"\n🚀 [3/3] predict.py 실행") 
    subprocess.run([
        sys.executable, "-m", "src.predict",
        "--name", args.name
    ], check=True)


def run_all(args):
    run_preprocess()
    run_train(args)
    run_predict(args)


if __name__ == "__main__":
    # ✅ 기본값 설정: 아무 인자도 없을 경우 자동 실행
    if len(sys.argv) == 1:
        print("⚙️ 기본값으로 실행합니다.")
        print("python -m main all --model yolov8x --name v8x --b 32 --e 50")
        sys.argv += ["all", "--model", "yolov8x", "--name", "v8x", "--b", "32", "--e", "50"]

    parser = argparse.ArgumentParser(description="YOLO 파이프라인 메인 실행 스크립트")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess
    subparsers.add_parser("preprocess", help="데이터 전처리 실행")

    # train
    train_parser = subparsers.add_parser("train", help="모델 학습 실행")
    train_parser.add_argument("--model", type=str, required=True)
    train_parser.add_argument("--name", type=str, required=True)
    train_parser.add_argument("--b", type=int, required=True)
    train_parser.add_argument("--e", type=int, required=True)

    # predict
    predict_parser = subparsers.add_parser("predict", help="예측 및 제출 파일 생성")
    predict_parser.add_argument("--name", type=str, required=True)

    # all-in-one 실행
    all_parser = subparsers.add_parser("all", help="전처리+학습+예측 전체 실행")
    all_parser.add_argument("--model", type=str, required=True)
    all_parser.add_argument("--name", type=str, required=True)
    all_parser.add_argument("--b", type=int, required=True)
    all_parser.add_argument("--e", type=int, required=True)

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess()
    elif args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "all":
        run_all(args)
