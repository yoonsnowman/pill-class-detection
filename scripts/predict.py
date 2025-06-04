# predict.py
from ultralytics import YOLO
import os
import argparse
import json
import re
import pandas as pd
import torch
from tqdm import tqdm # tqdm 임포트 추가
import configs.config_paths as cc

# --- 장치(Device) 설정 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 전역 설정 변수 (configs/config_paths.py 파일에서 로드) ---
BASE_OUTPUT_DIR = cc.OUTPUT_DIR
TEST_IMAGES_DIR = cc.TEST_IMG_DIR
CATEGORY_ID_MAP_PATH = cc.CAT_ID_DIR
SUBMISSION_FILENAME = cc.SUBMISSION_FILENAME

# --- 스크립트 내 기본값 설정 ---
MODEL_FILENAME_FROM_TRAIN = 'best.pt'
LOCAL_DEFAULT_CONF = 0.25
LOCAL_DEFAULT_IOU = 0.45
USE_TTA_PREDICT = False


def get_raw_predictions(model_load_path, conf_to_use, iou_to_use):
    """
    지정된 모델 경로와 Confidence, IoU 설정을 사용하여 테스트 이미지에 대한 예측을 수행합니다.
    TTA는 사용하지 않으며, 예측 결과를 내부 리스트 형태로 반환합니다. tqdm으로 진행 상황을 표시합니다.
    """
    print(f"\n✅ [예측 실행] 모델 로드 및 추론 시작 (Conf={conf_to_use}, IoU={iou_to_use}, TTA={USE_TTA_PREDICT})...")
    
    if not os.path.exists(model_load_path):
        print(f"⚠️ 예측에 사용할 모델 파일을 찾을 수 없습니다: {model_load_path}")
        return None

    model = YOLO(model_load_path)
    print(f"[예측 실행] 모델 로드 완료: {model_load_path}")

    # 테스트 이미지 개수 파악 (tqdm의 total 인자로 사용)
    try:
        # 일반 파일만 필터링 (하위 폴더 등 제외)
        test_image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if os.path.isfile(os.path.join(TEST_IMAGES_DIR, f))]
        num_test_images = len(test_image_files)
        if num_test_images == 0:
            print(f"⚠️ [예측 실행] 테스트 이미지 폴더에 이미지가 없습니다: {TEST_IMAGES_DIR}")
            return None
    except FileNotFoundError:
        print(f"⚠️ [예측 실행] 테스트 이미지 폴더를 찾을 수 없습니다: {TEST_IMAGES_DIR}")
        return None
    except Exception as e:
        print(f"⚠️ [예측 실행] 테스트 이미지 개수 파악 중 오류 발생: {e}")
        return None


    try:
        results_iterator = model.predict(
            source=TEST_IMAGES_DIR, imgsz=640, conf=conf_to_use, iou=iou_to_use,
            augment=USE_TTA_PREDICT,
            save=False,
            stream=True, # stream=True일 때 results_iterator는 generator
            device=DEVICE, verbose=False
        )
    except Exception as e:
        print(f"⚠️ [예측 실행] model.predict() 실행 중 오류 발생: {e}")
        return None

    all_img_predictions = []
    print(f"[예측 실행] 테스트 이미지 폴더: {TEST_IMAGES_DIR}, 총 {num_test_images}개 이미지")

    # tqdm으로 results_iterator를 감싸서 진행 상황 표시
    # desc: 진행 바 앞에 표시될 설명
    # total: 전체 작업량 (이미지 개수)
    # unit: 각 단계의 단위 (예: 'img' 또는 'it')
    # ncols: 진행 바의 너비 (자동 조절을 위해 None 또는 생략 가능)
    # bar_format: 진행 바 표시 형식 (기본값 사용 가능)
    for res in tqdm(results_iterator, total=num_test_images, desc="[예측 진행]", unit="img"):
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
    
    # tqdm 사용 시 루프 후 자동으로 줄바꿈이 되므로, 별도 print("\n")이 필요 없을 수 있음
    # 마지막 완료 메시지는 그대로 유지
    if len(all_img_predictions) == num_test_images : # 모든 이미지가 처리되었는지 간단히 확인 (오류 없이 끝났는지)
        print(f"[예측 실행] 총 {len(all_img_predictions)}개 이미지 처리 완료.")
    else: # 예측 결과 수와 실제 이미지 수가 다를 경우 (예: predict 중단)
        print(f"[예측 실행] {len(all_img_predictions)}개 이미지 처리됨 (총 {num_test_images}개 중).")


    if not all_img_predictions:
        print("⚠️ [예측 실행] 생성된 예측 결과가 없습니다.")
        return None
        
    return all_img_predictions

def generate_and_save_submission_csv(raw_predictions_data, submission_file_path):
    """
    가공된 예측 결과(raw_predictions_data)를 바탕으로 최종 제출용 CSV 파일을 생성합니다.
    """
    print("\n✅ [제출 파일 생성] 시작...")

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
                print(f"⚠️ Category ID 매핑에 없는 YOLO 클래스 ID 발견: {yolo_class_id} (이미지 ID: {image_submission_id}). 이 box는 건너<0xEB><0x81><0xB5>니다.")
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
        print(f"✅ [제출 파일 생성] CSV 파일 저장 완료: {submission_file_path} (총 {len(submission_df)}개 라벨)")
    except IOError as e:
        print(f"⚠️ [제출 파일 생성] CSV 파일 저장 실패: {e} (경로: {submission_file_path})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 모델 예측 및 제출 파일 생성 스크립트. 실행 폴더 이름을 필수로 지정해야 합니다.")
    parser.add_argument('--name', type=str, required=True,
                        help="예측에 사용할 모델이 있는 폴더 이름 (예: yolov8_run1). 'outputs' 폴더 내에 해당 이름의 폴더가 있어야 합니다.")
    
    args = parser.parse_args()

    current_run_folder_path = os.path.join(BASE_OUTPUT_DIR, args.name)
    model_for_prediction_path = os.path.join(current_run_folder_path, 'weights', MODEL_FILENAME_FROM_TRAIN)
    submission_csv_output_path = os.path.join(current_run_folder_path, SUBMISSION_FILENAME)

    print("--- 예측 및 제출 파일 생성 설정 ---")
    print(f"참조 실행 폴더 이름: {args.name}")
    print(f"사용 모델 경로: {model_for_prediction_path}")
    print(f"테스트 이미지 폴더: {TEST_IMAGES_DIR}")
    print(f"카테고리 ID 맵 경로: {CATEGORY_ID_MAP_PATH}")
    print(f"사용 장치: {DEVICE}")
    print(f"사용 Confidence: {LOCAL_DEFAULT_CONF}")
    print(f"사용 IoU: {LOCAL_DEFAULT_IOU}")
    print(f"TTA(테스트 시 증강) 사용 안 함: {not USE_TTA_PREDICT}")
    print(f"최종 제출 CSV 파일 출력 경로: {submission_csv_output_path}")
    print("----------------------------------")
    
    raw_predictions = get_raw_predictions(model_for_prediction_path, LOCAL_DEFAULT_CONF, LOCAL_DEFAULT_IOU)

    if raw_predictions:
        generate_and_save_submission_csv(raw_predictions, submission_csv_output_path)
    else:
        print("\n최종 예측 결과가 없어 제출 파일을 생성하지 않습니다.")