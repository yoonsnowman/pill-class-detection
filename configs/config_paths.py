# configs/config_paths.py
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ---------- preprocess.py 호출용 ----------
PRE_IN_DIR = 'data/yolo'
PRE_OUT_DIR = 'data/yolo/pill_yolo_format'


# ---------- oversample.py 호출용 ----------
TRAIN_IMG_DIR = 'data/yolo/pill_yolo_format/images/train/'
TRAIN_LB_DIR = 'data/yolo/pill_yolo_format/labels/train/'
# TEST_IMG_DIR  # predict.py에 존재하여 주석처리



# ---------- train.py 호출용 ----------
YAML_DIR = 'data/yolo/pill_yolo_format/data.yaml'  #YAML 파일 경로
OUTPUT_DIR = 'outputs'  # 예측값 출력 폴더


# ---------- predict.py 호출용 ----------
# OUTPUT_DIR = 'outputs'  # train.py에 존재하여 주석처리
TEST_IMG_DIR = 'data/yolo/pill_yolo_format/images/test/'  # test 이미지 폴더 경로
CAT_ID_DIR = 'data/yolo/pill_yolo_format/yolo_to_categoryid.json' # 카테고리 ID 파일 경로


# ---------- class_counter.py 호출용 ----------
TRAIN_LB_DIR = 'data/yolo/pill_yolo_format/labels/train/'
# PRE_OUT_DIR  # preprocess.py에 존재하여 주석처리
# CAT_ID_DIR  # predict.py에 존재하여 주석처리
# YAML_DIR  # train.py에 존재하여 주석처리





# ---------- class_counter.py 호출용 ----------
IMAGE_PATH = 'data/yolo/pill_yolo_format/images/'  # 이미지 폴더 경로
LABEL_PATH = 'data/yolo/pill_yolo_format/labels/'  # 라벨 폴더 경로


# ---------- 아직 미사용 ----------
VAL_IMG_DIR = 'data/yolo/pill_yolo_format/images/val/'  # val 이미지 폴더 경로
VAL_LB_DIR = 'data/yolo/pill_yolo_format/labels/val/'  # val 라벨 폴더 경로


# ---------- 시각화 호출용 ----------
FONT_DIR = 'data/font/NanumGothic.ttf'  # 나눔고딕



# ---------- 폰트 설정 함수 ----------
def setup_font():
    """지정된 경로의 나눔고딕 폰트를 matplotlib에 추가합니다."""
    if os.path.exists(FONT_DIR):
        # 폰트 매니저에 폰트 추가
        fm.fontManager.addfont(FONT_DIR)
        # matplotlib의 기본 폰트를 나눔고딕으로 설정
        plt.rcParams['font.family'] = 'NanumGothic'
    else:
        print(f"⚠️ 나눔고딕 폰트를 찾을 수 없습니다: {FONT_DIR}")
setup_font()