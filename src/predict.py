# predict.py
from ultralytics import YOLO
import os
import argparse
import json
import re
import pandas as pd
import torch
from tqdm import tqdm
import configs.config_paths as cc


# ---------- 전역 설정 변수 ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_OUTPUT_DIR = cc.OUTPUT_DIR
TEST_IMAGES_DIR = cc.TEST_IMG_DIR
CATEGORY_ID_MAP_PATH = cc.CAT_ID_DIR


# ---------- 예측 기본값 설정 ----------
MODEL_FILENAME_FROM_TRAIN = 'best.pt'
LOCAL_DEFAULT_CONF = 0.001
LOCAL_DEFAULT_IOU = 0.45


# ---------- 예측 함수 ----------
def get_raw_predictions(model_load_path, conf_to_use, iou_to_use):
    print(f"✅ 추론 시작 (Conf={conf_to_use}, IoU={iou_to_use})")
    
    if not os.path.exists(model_load_path):
        print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_load_path}")
        return None

    model = YOLO(model_load_path)

    # 테스트 이미지 개수 파악
    try:
        # 일반 파일만 필터링 (하위 폴더 등 제외)
        test_image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if os.path.isfile(os.path.join(TEST_IMAGES_DIR, f))]
        num_test_images = len(test_image_files)
        if num_test_images == 0:
            print(f"⚠️ 테스트 이미지 폴더에 이미지가 없습니다: {TEST_IMAGES_DIR}")
            return None
    except FileNotFoundError:
        print(f"⚠️ 테스트 이미지 폴더를 찾을 수 없습니다: {TEST_IMAGES_DIR}")
        return None
    except Exception as e:
        print(f"⚠️ 테스트 이미지 개수 파악 중 오류 발생: {e}")
        return None

    try:
        results_iterator = model.predict(
            source=TEST_IMAGES_DIR, imgsz=640, conf=conf_to_use, iou=iou_to_use,
            augment=False,
            save=False,
            stream=True,
            device=device, verbose=False
        )
    except Exception as e:
        print(f"⚠️ model.predict() 실행 중 오류 발생: {e}")
        return None

    all_img_predictions = []

    for res in tqdm(results_iterator, total=num_test_images, desc="테스트 데이터 로드", unit="img"):
        img_path = res.path
        img_filename = os.path.basename(img_path)
        match_id = re.search(r'(\d+)\.\w+', img_filename)
        img_id = int(match_id.group(1)) if match_id else img_filename

        current_img_preds = {"image_path": img_path, "image_id": img_id, "boxes": []}
        if res.boxes:
            for box_obj in res.boxes:
                current_img_preds["boxes"].append({
                    "xyxy": box_obj.xyxy[0].tolist(),
                    "conf": float(box_obj.conf[0]),
                    "cls": int(box_obj.cls[0])
                })
        all_img_predictions.append(current_img_preds)
    
    if not all_img_predictions:
        print("⚠️ [예측 실행] 생성된 예측 결과가 없습니다.")
        return None
        
    return all_img_predictions


# ---------- 제출용 CSV 파일 생성 함수 ----------
def generate_and_save_submission_csv(raw_predictions_data, submission_file_path):
    if not raw_predictions_data:
        print("⚠️ [제출 파일 생성] 입력된 예측 데이터가 없습니다. CSV 파일을 생성할 수 없습니다.")
        return

    if not os.path.exists(CATEGORY_ID_MAP_PATH):
        print(f"⚠️ Category ID 매핑 JSON 파일을 찾을 수 없습니다: {CATEGORY_ID_MAP_PATH}")
        return
        
    try:
        with open(CATEGORY_ID_MAP_PATH, 'r', encoding='utf-8') as f:
            yolo_cls_to_submission_cat_id = {int(k): int(v) for k, v in json.load(f).items()}
    except Exception as e:
        print(f"⚠️ Category ID 매핑 JSON 파일 로드 중 오류 발생: {e} (경로: {CATEGORY_ID_MAP_PATH})")
        return

    output_rows = []
    current_annotation_id = 1

    for single_image_prediction in raw_predictions_data:
        image_submission_id = single_image_prediction["image_id"]
        
        if not single_image_prediction["boxes"]:
            continue

        for detected_box in single_image_prediction["boxes"]:
            x1, y1, x2, y2 = detected_box["xyxy"]
            box_width = x2 - x1
            box_height = y2 - y1
            box_score = round(detected_box["conf"], 4)
            yolo_class_id = detected_box["cls"]

            if yolo_class_id not in yolo_cls_to_submission_cat_id:
                print(f"⚠️ Category ID 매핑에 없는 YOLO 클래스 ID 발견: {yolo_class_id} (이미지 ID: {image_submission_id}).")
                continue
            submission_category_id = yolo_cls_to_submission_cat_id[yolo_class_id]
            output_rows.append([
                current_annotation_id, image_submission_id, submission_category_id,
                int(round(x1)), int(round(y1)), int(round(box_width)), int(round(box_height)), box_score
            ])
            current_annotation_id += 1
    
    if not output_rows:
        print("⚠️ [제출 파일 생성] CSV로 변환할 예측 결과(bounding box)가 없습니다.")
        return

    submission_columns = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
    submission_df = pd.DataFrame(output_rows, columns=submission_columns)
    
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
    try:
        submission_df.to_csv(submission_file_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 생성 완료: {submission_file_path}")
    except IOError as e:
        print(f"⚠️ [제출 파일 생성] CSV 파일 저장 실패: {e} (경로: {submission_file_path})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 모델 예측 및 제출 파일 생성 스크립트. 실행 폴더 이름을 필수로 지정해야 합니다.")
    parser.add_argument('--name', type=str, required=True,
                        help="예측에 사용할 모델이 있는 폴더 이름 (예: yolov8_run1). 'outputs' 폴더 내에 해당 이름의 폴더가 있어야 합니다.")
    
    args = parser.parse_args()

    current_run_folder_path = os.path.join(BASE_OUTPUT_DIR, args.name)
    model_for_prediction_path = os.path.join(current_run_folder_path, 'weights', MODEL_FILENAME_FROM_TRAIN)
    submission_csv_output_path = os.path.join(current_run_folder_path, 'submission.csv')

    raw_predictions = get_raw_predictions(model_for_prediction_path, LOCAL_DEFAULT_CONF, LOCAL_DEFAULT_IOU)

    if raw_predictions:
        generate_and_save_submission_csv(raw_predictions, submission_csv_output_path)
    else:
        print("\n최종 예측 결과가 없어 제출 파일을 생성하지 않습니다.")