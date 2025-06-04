import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
from collections import Counter

# ────────────────────────────────
# 📁 경로 설정
label_dir     = 'data/yolo/pill_yolo_format/labels/train_aug'
json_path     = 'data/yolo/pill_yolo_format/yolo_to_categoryid.json'
yaml_path     = 'data/yolo/pill_yolo_format/data.yaml'
save_csv_path = 'data/yolo/pill_yolo_format/class_distribution_aug.csv'

# ────────────────────────────────
# 📥 파일 불러오기
with open(json_path, encoding='utf-8') as jf:
    yolo_to_categoryid = json.load(jf)

with open(yaml_path, encoding='utf-8') as yf:
    data_yaml = yaml.safe_load(yf)
    yolo_names = data_yaml['names']  # YOLO 클래스 ID → 이름

# ────────────────────────────────
# 🔍 클래스 등장 횟수 세기
class_counter = Counter()
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

print("\n📦 라벨 파일 분석 중:")
for file in tqdm(label_files):
    with open(os.path.join(label_dir, file)) as f:
        for line in f:
            if line.strip() == '':
                continue
            cls_id = int(line.strip().split()[0])
            class_counter[cls_id] += 1

# ────────────────────────────────
# 📊 결과 출력 + CSV 저장용 리스트 생성
print("\n🎯 클래스별 라벨 등장 횟수 (YOLO ID → Category ID):")
csv_rows = []
for cls_id in sorted(class_counter.keys()):
    count = class_counter[cls_id]
    category_id = yolo_to_categoryid[str(cls_id)]
    pill_name = yolo_names[cls_id]

    # 콘솔 출력 (요청 포맷)
    print(f"YOLO Class {cls_id:<2d} | Category ID {str(category_id):<5} | {count:<5}개 | {pill_name}")

    # CSV 행 저장
    csv_rows.append({
        'YOLO_class': cls_id,
        'Category_ID': category_id,
        'Pill_Name': pill_name,
        'Count': count
    })

# ────────────────────────────────
# 📁 CSV 저장
df = pd.DataFrame(csv_rows)
df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 클래스 통계 CSV 저장 완료: {save_csv_path}")
