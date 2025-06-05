# configs/config_paths.py
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 경로 설정 ---

# 데이터셋 구성 YAML 파일 경로
YAML_DIR = 'data/yolo/pill_yolo_format/data.yaml'

# train 이미지 폴더 경로
TRAIN_IMG_DIR = 'data/yolo/pill_yolo_format/images/train/'

# test 이미지 폴더 경로
TEST_IMG_DIR = 'data/yolo/pill_yolo_format/images/test/'

# val 이미지 폴더 경로
VAL_IMG_DIR = 'data/yolo/pill_yolo_format/images/val/'

# train 라벨 폴더 경로
TRAIN_LB_DIR = 'data/yolo/pill_yolo_format/labels/train/'

# val 라벨 폴더 경로
VAL_LB_DIR = 'data/yolo/pill_yolo_format/labels/val/'

# YOLO 클래스 ID와 제출용 카테고리 ID 매핑 파일 경로
CAT_ID_DIR = 'data/yolo/pill_yolo_format/yolo_to_categoryid.json'

# 시각화 등에 사용될 나눔고딕 폰트 파일 경로
FONT_DIR = 'data/font/NanumGothic.ttf'

# 모든 실행 결과(학습, 예측 등)가 저장될 기본 폴더
OUTPUT_DIR = 'outputs' # 최상위 출력 폴더

# 최종 제출용 CSV 파일 이름 (predict.py에서 사용)
SUBMISSION_FILENAME = 'submission.csv'

# path_test.py 호출용
TEST_PATH = 'data/yolo/pill_yolo_format/images/'



# --- 폰트 설정 함수 ---
def setup_font():
    """지정된 경로의 나눔고딕 폰트를 matplotlib에 추가합니다."""
    if os.path.exists(FONT_DIR):
        # 폰트 매니저에 폰트 추가
        fm.fontManager.addfont(FONT_DIR)
        # matplotlib의 기본 폰트를 나눔고딕으로 설정
        plt.rcParams['font.family'] = 'NanumGothic'
        print(f"✅ 나눔고딕 폰트 설정 완료: {FONT_DIR}")
    else:
        print(f"⚠️ 나눔고딕 폰트를 찾을 수 없습니다: {FONT_DIR}")

# 스크립트 로드 시 폰트 설정 실행
setup_font()