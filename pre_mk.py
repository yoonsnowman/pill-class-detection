import os
import sys
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def develop(environment='base'):
    n0, d0 = 'train_anno', 'superdog/train_annotations'
    n1, d1 = 'train',      'superdog/train_images'
    n2, d2 = 'temp',       ''
    n3, d3 = 'test',       'superdog/test_images'
    n4, d4 = 'font',       'superdog/NanumGothic.ttf'
    n5, d5 = 'labels',     'superdog/processed/labels'
    n6, d6 = 'classes',    'superdog/processed/classes.txt'
    dirs = {n0: d0, n1: d1, n2: d2, n3: d3, n4: d4, n5: d5, n6: d6}

    local_path = 'G:/내 드라이브/dev/'
    cloud_path = '/content/drive/MyDrive'
    env_path = f'{cloud_path}/dev/environment/{environment}'

    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        if env_path not in sys.path:
            sys.path.insert(0, env_path)
        os.environ['TORCH_HOME'] = f'{cloud_path}/dev/model/pytorch_model'
        return {k: os.path.join(cloud_path, v) for k, v in dirs.items()}
    else:
        return {k: os.path.join(local_path, v) for k, v in dirs.items()}
paths = develop()


import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw
import numpy as np

# ========================== 경로 설정 ==========================
#anno_root_dir = Path("/content/data/train_annotations")
#image_root_dir = Path("/content/data/train_images")
#output_base_path = Path("/content/yolo_dataset") 
anno_root_dir = Path("/content/drive/MyDrive/superdog/train_annotations")
image_root_dir = Path("/content/drive/MyDrive/superdog/train_images")
output_base_path = Path("/content/drive/MyDrive/superdog/yolo_dataset") 
output_base_path.mkdir(parents=True, exist_ok=True)

merged_json_path = output_base_path / "merged_coco.json"
image_output_dir = output_base_path / "images"
label_output_dir = output_base_path / "labels"

# ========================== 1. 어노테이션 병합 ==========================
# COCO 형식으로 병합할 데이터 구조 초기화하고, 
# 새로운 image/annotation id를 관리하기 위한 카운터 정의
# 카테고리 ID 매핑 및 클래스 이름 저장용 리스트 생성
# =========================================================================
merged = {"images": [], "annotations": [], "categories": []}
anno_id_counter = 1
img_id_counter = 1
original_cat_id_to_yolo_id = {}
yolo_class_names = []

# ------------------------ 카테고리 수집 ----------------------------------
# JSON 안에 하위 폴더가 여러개있으니깐 
# 모든 하위 폴더 내의 JSON 파일들을 돌면서, 각 JSON의 categories 항목을 수집하여 temp_coco_categories에 저장
# 중복 제거와 정렬을 통해 일관된 클래스 인덱스를 부여할 수 있도록 준비
# --------------------------------------------------------------------------
temp_coco_categories = {}

for folder in tqdm(sorted(anno_root_dir.iterdir()), desc="Collecting categories"):
    if not folder.is_dir():
        continue
    for subfolder in folder.iterdir():
        if not subfolder.is_dir():
            continue
        for json_file in subfolder.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for cat in data.get("categories", []):
                        temp_coco_categories[cat["id"]] = cat["name"]
                except:
                    continue

# ------------------------ category_id 매핑 ------------------------
# COCO의 기존 category ID를 YOLO 형식의 연속된 정수 ID로 매핑
# YOLO에 맞는 클래스 이름 순서를 위한 리스트 생성
# ------------------------------------------------------------------
sorted_cat_ids = sorted(temp_coco_categories.keys())

for new_id, old_id in enumerate(sorted_cat_ids):
    original_cat_id_to_yolo_id[old_id] = new_id
    yolo_class_names.append(temp_coco_categories[old_id])

image_file_name_set = set()
file_name_to_new_img_id = {}

# ------------------------- 이미지 및 어노테이션 병합 -------------------------
# JSON에서 images, annotations 필드를 돌면서 중복 없이 병합
# file_name 기준으로 중복 이미지 제거 및 새로운 ID 부여
# bbox는 유효한 값만 필터링 후, category_id를 YOLO ID로 변환하여 저장
# -----------------------------------------------------------------------------
for folder in tqdm(sorted(anno_root_dir.iterdir()), desc="Merging images and annotations"):
    if not folder.is_dir():
        continue
    for subfolder in folder.iterdir():
        if not subfolder.is_dir():
            continue
        for json_file in subfolder.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    continue
                if "images" not in data or "annotations" not in data:
                    continue

                current_json_img_id_map = {}

                for img in data["images"]:
                    file_name = img["file_name"]
                    if file_name in image_file_name_set:
                        current_json_img_id_map[img["id"]] = file_name_to_new_img_id[file_name]
                        continue

                    image_file_name_set.add(file_name)
                    file_name_to_new_img_id[file_name] = img_id_counter
                    current_json_img_id_map[img["id"]] = img_id_counter

                    merged["images"].append({
                        "id": img_id_counter,
                        "file_name": file_name,
                        "width": img["width"],
                        "height": img["height"]
                    })
                    img_id_counter += 1

                for anno in data["annotations"]:
                    bbox = anno.get("bbox")
                    if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                        continue

                    original_image_id = anno["image_id"]
                    new_image_id = current_json_img_id_map.get(original_image_id)
                    if new_image_id is None:
                        continue

                    yolo_cat_id = original_cat_id_to_yolo_id.get(anno["category_id"])
                    if yolo_cat_id is None:
                        continue

                    merged["annotations"].append({
                        "id": anno_id_counter,
                        "image_id": new_image_id,
                        "category_id": yolo_cat_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    })
                    anno_id_counter += 1

merged["categories"] = [
    {"id": i, "name": name, "supercategory": "pill"}
    for i, name in enumerate(yolo_class_names)
]

with open(merged_json_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)

print(f"병합 완료: {len(merged['images'])} 이미지, {len(merged['annotations'])} 어노테이션")

# ========================== 2. Train/Val 분할 =======================================
# 20%를 validation set으로 분할하고, image_id를 기준으로 train/val ID 집합을 나눔
# ====================================================================================
val_ratio = 0.2

image_id_to_file = {img["id"]: img["file_name"] for img in merged["images"]}
all_ids = list(image_id_to_file.keys())

# 무작위로 섞기위해 
random.seed(42)
random.shuffle(all_ids)

val_ids = set(all_ids[:int(len(all_ids) * val_ratio)])            # 전체 이미지 ID 중 앞의 20%를 검증용 이미지 ID
train_ids = set(all_ids[int(len(all_ids) * val_ratio):])          # 나머지 80%를 학습용

image_info_map = {img["id"]: img for img in merged["images"]}     # ID로 빠르게 이미지 정보를 찾을 수 있도록 image_info_map에 담기 

# ========================== 3. 이미지 리사이즈 및 라벨 저장 ==========================
# 이미지 파일을 열고 640x640 사이즈로 리사이즈 후 저장 => 복사하면 기존 사이즈로 가져와서 복사 아닌 저장 
# 각 annotation에 대해 YOLO 형식(class_id x_center y_center width height)의 라벨을 텍스트 파일로 저장
# 정규화 좌표를 계산하고, bbox가 이미지 경계를 벗어나지 않도록 clipping 처리 => 파일 생성 시 bbox 오류가 발생하여
# =====================================================================================
resize_size = 640

for split, ids in [("train", train_ids), ("val", val_ids)]:
    (image_output_dir / split).mkdir(parents=True, exist_ok=True)   # parents=True : 상위 디렉토리가 존재하지 않으면 자동으로 생성해줌 
    (label_output_dir / split).mkdir(parents=True, exist_ok=True)   # exist_ok=True : 이미 디렉토리가 존재하면 오류를 발생시키지 않고 넘어감 

    print(f"Processing {split} images")
    for img_id in tqdm(ids):
        img_info = image_info_map[img_id]
        src = image_root_dir / img_info["file_name"]                # 원본 이미지가 저장될 경로
        dst = image_output_dir / split / img_info["file_name"]      # 리사이즈된 이미지가 저장될 경로
        if not src.exists():                                        # 파일 없으면 건너뛰기 
            continue
        with Image.open(src) as im:
            im_resized = im.resize((resize_size, resize_size))      # 이미지를 640,640으로 리사이즈 해줌 
            im_resized.save(dst)

    print(f"Generating {split} labels")

    for anno in merged["annotations"]:                              # 여기가 라벨 생성 -> 없으면 계속
        if anno["image_id"] not in ids:
            continue
        img_info = image_info_map[anno["image_id"]]
        x, y, bw, bh = anno["bbox"]

        # 비율 계산
        scale_x = resize_size / img_info["width"]
        scale_y = resize_size / img_info["height"]

        # bbox 리사이즈                                             # 이미지를 리사이즈 하니깐 bbox가 리사이즈된 이미지를 벗어나지 않도록 잘라줌 
        x *= scale_x
        y *= scale_y
        bw *= scale_x
        bh *= scale_y

        # 이미지 경계 안으로 clipping          
        x = max(0, min(x, resize_size - 1))
        y = max(0, min(y, resize_size - 1))
        bw = max(1, min(bw, resize_size - x))
        bh = max(1, min(bh, resize_size - y))

        # 정규화
        x_center = (x + bw / 2) / resize_size
        y_center = (y + bh / 2) / resize_size
        norm_w = bw / resize_size
        norm_h = bh / resize_size

        # x_center, y_center는 0 이상 1 이하, norm_w, norm_h는 0보다 크고 1 이하가 아니면 
        # 어떤 문제인지 확인하기 위해 로그를 남겨 띄워줌 
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < norm_w <= 1 and 0 < norm_h <= 1):
            print(f"경고: 정규화 bbox 오류 - {img_info['file_name']}: {(x_center, y_center, norm_w, norm_h)}")
            continue

        label_path = label_output_dir / split / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_path, 'a') as f:
            f.write(f"{anno['category_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

# ========================== 4. data.yaml 생성 ==========================
# YOLO 학습 시 필요한 data.yaml 파일 생성
# =======================================================================
yaml_path = output_base_path / "data.yaml"
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"path: {output_base_path}\n")
    f.write(f"train: images/train\n")
    f.write(f"val: images/val\n")
    f.write(f"names: {json.dumps(yolo_class_names, ensure_ascii=False)}\n")
    f.write(f"nc: {len(yolo_class_names)}\n")

print(f" 변환 및 data.yaml 저장: {yaml_path}")



# ===================== 시각화 =====================
# ========================== 경로 설정 ==========================
image_dir = "/content/yolo_dataset/images/train/"
label_dir = "/content/yolo_dataset/labels/train/"
data_yaml_path = "/content/yolo_dataset/data.yaml"

# ========================== 폰트 설치 및 설정 ==========================
# Colab에 나눔고딕 폰트 설치 (런타임이 재시작 시 다시 실행)
#!sudo apt-get update > /dev/null 2>&1
#!sudo apt-get install -y fonts-nanum > /dev/null 2>&1
#!sudo fc-cache -fv > /dev/null 2>&1

# 설치된 폰트 경로 확인 및 설정
try:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    print(f"경고: 폰트 파일 '{font_path}'을(를) 찾을 수 없습니다. 기본 폰트 사용합니다.")
    font = ImageFont.load_default()

# 이미지 여러개 보여주기 시각화 
import math

num_images_to_visualize = 9  # 9개 이미지로 변경

# 이미지 최대 9개 추출
if len(all_image_files) > num_images_to_visualize:
    selected_image_files = random.sample(all_image_files, num_images_to_visualize)
else:
    selected_image_files = all_image_files

if not selected_image_files:
    print(f"'{image_dir}' 경로에 시각화할 이미지 파일이 없습니다.")
else:
    cols = 3
    rows = math.ceil(len(selected_image_files) / cols)

    plt.figure(figsize=(cols * 6, rows * 6))  # 가로 3, 세로 3 이미지, 각 이미지 6인치 크기

    for idx, img_filename in enumerate(selected_image_files):
        img_path = os.path.join(image_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        image_np = cv2.imread(img_path)
        if image_np is None:
            print(f"경로 '{img_path}'에서 이미지를 읽을 수 없습니다.")
            continue

        image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        h, w, _ = image_np_rgb.shape

        image_pil = Image.fromarray(image_np_rgb)
        draw = ImageDraw.Draw(image_pil)

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                try:
                    cls_id, x_center, y_center, norm_bw, norm_bh = map(float, line.strip().split())
                    cls_id = int(cls_id)

                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown_cls:{cls_id}"

                    x1 = int((x_center - norm_bw / 2) * w)
                    y1 = int((y_center - norm_bh / 2) * h)
                    x2 = int((x_center + norm_bw / 2) * w)
                    y2 = int((y_center + norm_bh / 2) * h)

                    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

                    text_x = x1
                    text_y = y1 - font_size - 5
                    if text_y < 0:
                        text_y = y1 + 5

                    draw.text((text_x, text_y), class_name, font=font, fill=(255, 0, 0))

                except (ValueError, IndexError) as e:
                    print(f"라벨 파일 '{label_path}' 파싱 오류 또는 클래스 ID 범위 오류: {line.strip()}. 오류: {e}")
                    continue
        else:
            print(f"라벨 파일이 존재하지 않거나 비어 있습니다: {label_path}")

        final_image_display = np.array(image_pil)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(final_image_display)
        plt.axis('off')
        plt.title(f"YOLO Bounding Boxes\n{img_filename}")

    plt.tight_layout()
    plt.show()

# 실행예시 
# yolo train model=yolov8n.pt data=/content/drive/MyDrive/superdog/yolo_dataset/data.yaml imgsz=640 epochs=100

# ================= TEST Data =================
# 입력 test 이미지 경로 (원본)
test_image_input_dir = Path("/content/drive/MyDrive/superdog/test_images")

# YOLO 구조 하위 경로
test_image_output_dir = Path("/content/drive/MyDrive/superdog/yolo_dataset/images/test")
test_image_output_dir.mkdir(parents=True, exist_ok=True)

resize_size = (640, 640)  # YOLO input size

# 이미지 복사 및 리사이즈
for img_path in tqdm(sorted(test_image_input_dir.glob("*.*")), desc="Processing test images"):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(resize_size, Image.LANCZOS)  # 고화질 리사이즈
        output_path = test_image_output_dir / img_path.name
        img.save(output_path)
    except Exception as e:
        print(f" 실패: {img_path.name} - {e}")

# 실행 방법
# python detect.py --weights runs/train/exp/weights/best.pt --source /content/yolo_dataset/images/test --img 640