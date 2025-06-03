import os
import sys
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def develop(environment='base'):
    """
    Sets up paths for local or Colab environment.
    """
    n0, d0 = 'train_anno', 'superdog/train_annotations'
    n1, d1 = 'train_images', 'superdog/train_images'
    n2, d2 = 'temp',       '' # Not used in this script, but kept for consistency
    n3, d3 = 'test_images', 'superdog/test_images'
    n4, d4 = 'font',       'superdog/NanumGothic.ttf' # Example font path
    n5, d5 = 'labels',     'superdog/processed/labels' # Not directly used, but implied by output_base_path
    n6, d6 = 'classes',    'superdog/processed/classes.txt' # Not directly used, but implied by output_base_path
    
    # Define base directories relative to which other paths are constructed
    # This structure makes it easier to manage paths
    dirs = {
        'train_annotations': d0, 
        'train_images': d1, 
        'test_images': d3, 
        'font': d4,
        'output_base': 'superdog/yolo_dataset' # Central output directory
    }

    local_base_path = 'G:/내 드라이브/' # Local base path
    cloud_base_path = '/content/drive/MyDrive' # Colab base path
    
    # Determine the base path based on the environment
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = cloud_base_path
        # Example for setting TORCH_HOME if needed
        os.environ['TORCH_HOME'] = f'{cloud_base_path}/dev/model/pytorch_model' 
    else:
        base_path = local_base_path

    # Construct full paths
    full_paths = {key: Path(base_path) / value for key, value in dirs.items()}
    return full_paths

paths = develop()

# ======================== 1. 어노테이션 병합 ========================
# COCO 형식 어노테이션 JSON 파일을 하나로 병합
# 카테고리 ID를 YOLO에 적합한 0부터 시작하는 ID로 재매핑
# ====================================================

# anno_root_dir: JSON 어노테이션이 포함된 디렉터리의 최상위 경로 (Path 객체)
# merged_json_path: 병합된 결과를 저장할 JSON 파일 경로
def merge_annotations(anno_root_dir, merged_json_path):
    merged = {"images": [], "annotations": [], "categories": []}
    anno_id_counter = 1
    img_id_counter = 1
    original_cat_id_to_yolo_id = {}
    yolo_class_names = []

    temp_coco_categories = {}

# ------------------------ 카테고리 수집 ----------------------------------
# JSON 안에 하위 폴더가 여러개있으니깐 
# 모든 하위 폴더 내의 JSON 파일들을 돌면서, 각 JSON의 categories 항목을 수집하여 temp_coco_categories에 저장
# 중복 제거와 정렬을 통해 일관된 클래스 인덱스를 부여할 수 있도록 준비
# JSON 파싱 오류가 있으면 경고 출력 후 건너뜀.
# --------------------------------------------------------------------------
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
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {json_file}")
                        continue
                    except Exception as e:
                        print(f"Error processing categories in {json_file}: {e}")
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
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {json_file}")
                        continue
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
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

    print(f"Merged {len(merged['images'])} images and {len(merged['annotations'])} annotations into {merged_json_path}")
    return merged, yolo_class_names

# ========================== 2. Train/Val 분할 =======================================
# 20%를 validation set으로 분할하고, image_id를 기준으로 train/val ID 집합을 나눔
# ====================================================================================
def split_dataset_and_process_images(merged_data, image_root_dir, image_output_dir, label_output_dir, resize_size=640, val_ratio=0.2):

    image_id_to_file = {img["id"]: img["file_name"] for img in merged_data["images"]}
    all_ids = list(image_id_to_file.keys())
    
    # 무작위로 섞기위해
    random.seed(42)
    random.shuffle(all_ids)

    val_ids = set(all_ids[:int(len(all_ids) * val_ratio)])              # 전체 이미지 ID 중 앞의 20%를 검증용 이미지 ID
    train_ids = set(all_ids[int(len(all_ids) * val_ratio):])            # 나머지 80%를 학습용

    image_info_map = {img["id"]: img for img in merged_data["images"]}  # ID로 빠르게 이미지 정보를 찾을 수 있도록 image_info_map에 담기 

# ========================== 3. 이미지 리사이즈 및 라벨 저장 ==========================
# 이미지 파일을 열고 640x640 사이즈로 리사이즈 후 저장 => 복사하면 기존 사이즈로 가져와서 복사 아닌 저장 
# 각 annotation에 대해 YOLO 형식(class_id x_center y_center width height)의 라벨을 텍스트 파일로 저장
# 정규화 좌표를 계산하고, bbox가 이미지 경계를 벗어나지 않도록 clipping 처리 => 파일 생성 시 bbox 오류가 발생하여
# =====================================================================================
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        (image_output_dir / split).mkdir(parents=True, exist_ok=True)    # parents=True : 상위 디렉토리가 존재하지 않으면 자동으로 생성해줌 
        (label_output_dir / split).mkdir(parents=True, exist_ok=True)    # exist_ok=True : 이미 디렉토리가 존재하면 오류를 발생시키지 않고 넘어감 

        print(f"Processing {split} images...")
        for img_id in tqdm(ids):
            img_info = image_info_map[img_id]
            src = image_root_dir / img_info["file_name"]                 # 원본 이미지가 저장될 경로
            dst = image_output_dir / split / img_info["file_name"]       # 리사이즈된 이미지가 저장될 경로
            if not src.exists():                                         # 파일 없으면 건너뛰기 
                print(f"Warning: Image file not found, skipping: {src}")
                continue
            try:
                with Image.open(src) as im:
                    im_resized = im.resize((resize_size, resize_size), Image.LANCZOS) # 이미지를 리사이즈 해줌 
                    im_resized.save(dst)
            except Exception as e:
                print(f"Error processing image {src}: {e}")
                continue

        print(f"Generating {split} labels...")
        for anno in tqdm(merged_data["annotations"], desc=f"Generating {split} labels"):
            if anno["image_id"] not in ids:
                continue
            img_info = image_info_map[anno["image_id"]]
            x, y, bw, bh = anno["bbox"]

            # 비율 계산
            scale_x = resize_size / img_info["width"]
            scale_y = resize_size / img_info["height"]

            # bbox 리사이즈                            # 이미지를 리사이즈 하니깐 bbox가 리사이즈된 이미지를 벗어나지 않도록 잘라줌 
            x_resized = x * scale_x
            y_resized = y * scale_y
            bw_resized = bw * scale_x
            bh_resized = bh * scale_y

            # 이미지 경계 안으로 clipping 
            x_clipped = max(0, min(x_resized, resize_size - 1))
            y_clipped = max(0, min(y_resized, resize_size - 1))
            
            
            bw_clipped = max(1, min(bw_resized, resize_size - x_clipped))
            bh_clipped = max(1, min(bh_resized, resize_size - y_clipped))
            
            # 정규화
            x_center = (x_clipped + bw_clipped / 2) / resize_size
            y_center = (y_clipped + bh_clipped / 2) / resize_size
            norm_w = bw_clipped / resize_size
            norm_h = bh_clipped / resize_size

            # x_center, y_center는 0 이상 1 이하, norm_w, norm_h는 0보다 크고 1 이하가 아니면 
            # 어떤 문제인지 확인하기 위해 로그를 남겨 띄워줌
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < norm_w <= 1 and 0 < norm_h <= 1):
                print(f"Warning: Normalized bbox out of bounds for {img_info['file_name']}: "
                      f"{(x_center, y_center, norm_w, norm_h)} from original bbox {anno['bbox']} "
                      f"on image size ({img_info['width']}, {img_info['height']}). Clipping applied: "
                      f"{(x_clipped, y_clipped, bw_clipped, bh_clipped)}")
                continue

            label_path = label_output_dir / split / f"{Path(img_info['file_name']).stem}.txt"
            with open(label_path, 'a') as f:
                f.write(f"{anno['category_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print("Image resizing and label generation complete.")

# ========================== 4. data.yaml 생성 ==========================
# YOLO 학습 시 필요한 data.yaml 파일 생성
# =======================================================================
def create_data_yaml(output_base_path, yolo_class_names):
    
    yaml_path = output_base_path / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {output_base_path.as_posix()}\n") # Use .as_posix() for platform-independent paths
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"names: {json.dumps(yolo_class_names, ensure_ascii=False)}\n")
        f.write(f"nc: {len(yolo_class_names)}\n")
    print(f"data.yaml created at: {yaml_path}")
    return yaml_path

# ========================== 5. 바운딩 박스 중복 방지 ==========================
# 치는 바운딩 박스 중에서 가장 적합한 하나만 남기고 나머지를 제거
# =======================================================================
def apply_nms_to_boxes(boxes, scores, iou_threshold=0.5):
    if not boxes:
        return []

    x1 = np.array([b[0] for b in boxes])
    y1 = np.array([b[1] for b in boxes])
    x2 = np.array([b[2] for b in boxes])
    y2 = np.array([b[3] for b in boxes])
    scores = np.array(scores)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]             # 뢰도 점수를 기준으로 바운딩 박스들의 인덱스를 내림차순으로 정렬

    keep = []                                  # 최종적으로 선택된 (유지될) 바운딩 박스의 인덱스를 저장할 리스트
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 겹치는 영역의 좌상단 x 좌표 중 더 큰 값을 찾기
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)      # 겹치는 영역의 너비를 계산
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# ========================== 6. 바운딩 박스 시각화 ==========================
# YOLO 형식으로 준비된 이미지와 라벨 데이터를 시각화
# =======================================================================
def visualize_yolo_data(image_dir, label_dir, class_names, font_path, num_images_to_visualize=9):
    
    # 코랩이면 한글 폰트 설치하기 위해
    if 'google.colab' in sys.modules:
        os.system('sudo apt-get update > /dev/null 2>&1')
        os.system('sudo apt-get install -y fonts-nanum > /dev/null 2>&1')
        os.system('sudo fc-cache -fv > /dev/null 2>&1')
        print("NanumGothic font installed (if not already present).")

    try:
        font_size = 20
        font = ImageFont.truetype(str(font_path), font_size) # Ensure font_path is string
    except IOError:
        print(f"Warning: Font file '{font_path}' not found. Using default font.")
        font = ImageFont.load_default()
    except Exception as e:
        print(f"Error loading font: {e}. Using default font.")
        font = ImageFont.load_default()

    # 지원되는 확장자의 이미지 파일 목록을 수집 후, 무작위로 최대 num_images_to_visualize 개 선택
    all_image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if len(all_image_files) > num_images_to_visualize:
        selected_image_files = random.sample(all_image_files, num_images_to_visualize)
    else:
        selected_image_files = all_image_files

    if not selected_image_files:
        print(f"No image files found in '{image_dir}' for visualization.")
        return

    cols = 3
    rows = math.ceil(len(selected_image_files) / cols)
    plt.figure(figsize=(cols * 6, rows * 6))

    for idx, img_filename in enumerate(selected_image_files):
        img_path = os.path.join(image_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        image_np = cv2.imread(img_path)
        if image_np is None:
            print(f"Could not read image from '{img_path}'.")
            continue

        image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)        # PIL에서 사용할 수 있도록 RGB로 변환
        h, w, _ = image_np_rgb.shape

        image_pil = Image.fromarray(image_np_rgb)
        draw = ImageDraw.Draw(image_pil)

        detected_boxes = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                try:
                    cls_id, x_center, y_center, norm_bw, norm_bh = map(float, line.strip().split())
                    cls_id = int(cls_id)

                    # 정규화된 좌표(0~1)를 원본 이미지 픽셀 좌표로 변환
                    x1 = int((x_center - norm_bw / 2) * w)
                    y1 = int((y_center - norm_bh / 2) * h)
                    x2 = int((x_center + norm_bw / 2) * w)
                    y2 = int((y_center + norm_bh / 2) * h)
                    
                    
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)

                    if x2 <= x1 or y2 <= y1: 
                        continue

                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown_cls:{cls_id}"
                    detected_boxes.append({"bbox": [x1, y1, x2, y2], "class_name": class_name, "score": 1.0}) # Dummy score for NMS

                except (ValueError, IndexError) as e:
                    print(f"Label file '{label_path}' parsing or class ID range error: {line.strip()}. Error: {e}")
                    continue
        else:
            print(f"Label file does not exist or is empty: {label_path}")

        
        if detected_boxes:
            boxes_for_nms = [d["bbox"] for d in detected_boxes]
            scores_for_nms = [d["score"] for d in detected_boxes]
            
            # NMS 적용
            keep_indices = apply_nms_to_boxes(boxes_for_nms, scores_for_nms, iou_threshold=0.3) # Adjust as needed
            
            for i in keep_indices:
                box_info = detected_boxes[i]
                x1, y1, x2, y2 = box_info["bbox"]
                class_name = box_info["class_name"]

                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

                text_x = x1
                text_y = y1 - font_size - 5
                if text_y < 0:
                    text_y = y1 + 5

                draw.text((text_x, text_y), class_name, font=font, fill=(255, 0, 0))

        final_image_display = np.array(image_pil)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(final_image_display)
        plt.axis('off')
        plt.title(f"YOLO Bounding Boxes\n{img_filename}")

    plt.tight_layout()
    plt.show()

# ========================== 7. Test Data 복사 ==========================
# Test 데이터 리사이즈 하면서 복사하여 저장
# =======================================================================
def process_test_images(test_image_input_dir, test_image_output_dir, resize_size=(640, 640)):
    
    test_image_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing test images from {test_image_input_dir} to {test_image_output_dir}...")
    for img_path in tqdm(sorted(test_image_input_dir.glob("*.*")), desc="Processing test images"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(resize_size, Image.LANCZOS)
            output_path = test_image_output_dir / img_path.name
            img.save(output_path)
        except Exception as e:
            print(f"Failed to process test image: {img_path.name} - {e}")
    print("Test image processing complete.")

# ========================== 8. 메인 ==========================
# 메인 실행
# =======================================================================
if __name__ == '__main__':
    
    anno_root_dir = paths['train_annotations']
    image_root_dir = paths['train_images']
    test_image_input_dir = paths['test_images']
    font_path_for_vis = paths['font'] 

    output_base_path = paths['output_base']
    output_base_path.mkdir(parents=True, exist_ok=True)

    merged_json_path = output_base_path / "merged_coco.json"
    image_output_dir = output_base_path / "images"
    label_output_dir = output_base_path / "labels"

    # 1. JSON 어노테이션을 하나로 병합
    print("Step 1: Merging annotations...")
    merged_data, yolo_class_names = merge_annotations(anno_root_dir, merged_json_path)

    # 2. 학습 데이터셋 분할 및 YOLO 포맷 라벨 생성
    print("\nStep 2: Splitting dataset and processing images/labels...")
    split_dataset_and_process_images(merged_data, image_root_dir, image_output_dir, label_output_dir)

    # 3. data.yaml 파일 생성
    print("\nStep 3: Creating data.yaml...")
    data_yaml_file = create_data_yaml(output_base_path, yolo_class_names)

    # 4. 시각화
    print("\nStep 4: Visualizing sample YOLO data...")
    visualize_yolo_data(image_output_dir / "train", label_output_dir / "train", yolo_class_names, font_path_for_vis)

    # 5. 테스트 이미지 전처리
    print("\nStep 5: Processing test images...")
    process_test_images(test_image_input_dir, image_output_dir / "test")

    print("\nData preprocessing complete. You can now train your YOLO model.")
    print(f"Example YOLO training command: yolo train model=yolov8n.pt data={data_yaml_file.as_posix()} imgsz=640 epochs=100")
    print(f"Example YOLO detection command: python detect.py --weights runs/train/exp/weights/best.pt --source {image_output_dir.as_posix()}/test --img 640")