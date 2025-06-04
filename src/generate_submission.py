from ultralytics import YOLO
import os
import pandas as pd
import re
import json

# ---- 1. 경로 설정 ----
model_path = 'data/yolo/run/yolo11x_run3_32_50_aug/weights/best.pt'
test_image_dir = 'data/yolo/pill_yolo_format/images/test/'
submission_csv_path = 'submission_yolo11x_run3_32_50_aug.csv'
categoryid_json_path = 'data/yolo/pill_yolo_format/yolo_to_categoryid.json'

# ---- 2. 모델 로드 ----
model = YOLO(model_path)

# ---- 3. 추론 수행 ----
results = model.predict(
    source=test_image_dir,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save=False,
    stream=True  # 메모리 상에서 순차 처리
)

# ---- 4. 결과 파싱 ----
submission_rows = []
annotation_id = 1

with open(categoryid_json_path, 'r', encoding='utf-8') as f:
    yolo_to_dl_idx = {int(k): int(v) for k, v in json.load(f).items()}

for r in results:
    file_name = os.path.basename(r.path)
    match = re.search(r'(\d+)', file_name)
    if not match:
        print(f"⚠️ 이미지 파일명에서 숫자 추출 실패: {file_name}")
        continue

    image_id = int(match.group(1))

    if r.boxes is None or len(r.boxes) == 0:
        continue

    for box in r.boxes:
        # box.xywh → (x_center, y_center, w, h), box.conf → conf, box.cls → class index
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 좌상단 x1, y1 / 우하단 x2, y2
        w = x2 - x1
        h = y2 - y1
        score = float(box.conf[0])
        category_id = yolo_to_dl_idx[int(box.cls[0])] #  + 1 YOLO 클래스 ID

        submission_rows.append([
            annotation_id, image_id, category_id,
            int(x1), int(y1), int(w), int(h), round(score, 4)
        ])
        annotation_id += 1

# ---- 5. DataFrame 생성 및 저장 ----
columns = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
df = pd.DataFrame(submission_rows, columns=columns)
df.to_csv(submission_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ 제출용 CSV 저장 완료: {submission_csv_path} (총 {len(df)}개 라벨)")
