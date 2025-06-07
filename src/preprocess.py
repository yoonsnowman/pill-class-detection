import json
import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
from collections import Counter
from tqdm import tqdm
import yaml
import configs.config_paths as cc


# 전역 설정 변수
input_dir = cc.PRE_IN_DIR
output_dir = cc.PRE_OUT_DIR


# ==================== COCO → YOLO 변환 함수 정의 ====================
# 여러 COCO 형식 JSON을 합치고 YOLO 형식으로 변환
# 학습/검증 데이터를 분리해 폴더로 저장
# 클래스 매핑 정보를 반환
# ==================================================================
def merge_and_convert_coco_to_yolo(json_paths, image_base_dir, output_base_dir, val_split=0.2):
    # -------------------- COCO 어노테이션 통합 --------------------
    yolo_labels_dir = os.path.join(output_base_dir, 'labels')
    yolo_images_dir = os.path.join(output_base_dir, 'images')

    os.makedirs(os.path.join(yolo_labels_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_labels_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_images_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_images_dir, 'val'), exist_ok=True)

    all_images = []
    all_annotations = []
    all_categories_info = []
    category_id_map = {}
    next_yolo_class_id = 0

    for json_path in tqdm(json_paths, desc="(Stage 1/5) JSON 어노테이션 통합"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'categories' in data and data['categories']:
                for category in data['categories']:
                    if category.get('id') is not None and category['id'] not in category_id_map:
                        category_id_map[category['id']] = next_yolo_class_id
                        all_categories_info.append({
                            'id': category['id'],
                            'name': category.get('name', f"class_{category['id']}"),
                            'yolo_id': next_yolo_class_id
                        })
                        next_yolo_class_id += 1

            if 'images' in data:
                # 이미지 정보가 존재하면 리스트에 추가
                all_images.extend(data['images'])

            if 'annotations' in data:
                # 어노테이션 정보가 존재하면 리스트에 추가
                all_annotations.extend(data['annotations'])
        except Exception:
            # 오류 발생 시 해당 파일 건너뛰기
            pass

    if not all_categories_info:
        return None, None

    final_class_names = [cat['name'] for cat in sorted(all_categories_info, key=lambda x: x['yolo_id'])]

    # 이미지 ID별 정보 구성
    image_info_map = {}
    for img in all_images:
        try:
            img_id = int(img['id'])
            image_info_map[img_id] = {
                'file_name': img.get('file_name'),
                'width': img.get('width'),
                'height': img.get('height')
            }
        except (ValueError, TypeError):
            # 잘못된 ID 형식은 건너뜀
            continue

    # 이미지 ID별 어노테이션 구성
    image_annotations_map = {}
    for anno in all_annotations:
        image_id_raw = anno.get('image_id')
        try:
            image_id = int(image_id_raw)
            if image_id not in image_annotations_map:
                image_annotations_map[image_id] = []
            image_annotations_map[image_id].append(anno)
        except (ValueError, TypeError):
            continue

    # 학습/검증 분할 대상 이미지 ID만 추출
    image_ids_to_split = [
        img_id for img_id in image_annotations_map.keys()
        if img_id in image_info_map and image_info_map[img_id].get('file_name')
    ]
    if not image_ids_to_split:
        return None, None

    # -------------------- COCO 어노테이션 통합 끝 --------------------
    # 필요한 이미지 및 어노테이션 정보를 모두 확보했으므로
    # 이제 Stratified Split을 수행하기 위한 준비가 완료되었습니다.

    # Stratified Split을 위한 라벨 추출: 각 이미지에 대해 첫 번째 어노테이션의 카테고리 사용
    stratify_labels_raw = []
    for img_id in image_ids_to_split:
        annotations_for_img = image_annotations_map.get(img_id)
        if annotations_for_img and annotations_for_img[0].get('category_id') is not None:
            coco_cat_id = annotations_for_img[0]['category_id']
            stratify_labels_raw.append(category_id_map.get(coco_cat_id))
        else:
            stratify_labels_raw.append(None)

    # -------------------- Train/Val Split 수행 --------------------
    valid_image_indices = [i for i, label in enumerate(stratify_labels_raw) if label is not None]
    if not valid_image_indices:
        # 모든 이미지에 대해 라벨 무시, 단순 랜덤 분할
        train_ids, val_ids = train_test_split(
            np.array(image_ids_to_split), test_size=val_split, random_state=42
        )
    else:
        image_ids_to_stratify = np.array(image_ids_to_split)[valid_image_indices]
        labels_for_stratify = np.array([l for l in stratify_labels_raw if l is not None])

        stratify_possible = True
        if len(set(labels_for_stratify)) < 2:
            stratify_possible = False
        else:
            class_counts = Counter(labels_for_stratify)
            if any(count < 2 for count in class_counts.values()):
                stratify_possible = False

        if stratify_possible:
            # 클래스 분포를 반영한 계층적 분할
            train_ids_strat, val_ids_strat = train_test_split(
                image_ids_to_stratify,
                test_size=val_split,
                random_state=42,
                stratify=labels_for_stratify
            )
            train_ids = list(train_ids_strat)
            val_ids = list(val_ids_strat)

            # 라벨이 없는 나머지 이미지는 별도로 랜덤 분할
            remaining_ids = [
                img_id for i, img_id in enumerate(image_ids_to_split)
                if i not in valid_image_indices
            ]
            if remaining_ids:
                train_rem, val_rem = train_test_split(
                    np.array(remaining_ids), test_size=val_split, random_state=42
                )
                train_ids.extend(list(train_rem))
                val_ids.extend(list(val_rem))
        else:
            # 라벨 분포가 불충분할 경우, 전체 이미지 랜덤 분할
            train_ids, val_ids = train_test_split(
                np.array(image_ids_to_split), test_size=val_split, random_state=42
            )

    train_ids = list(train_ids)
    val_ids = list(val_ids)

    # -------------------- YOLO 변환 및 이미지 복사 --------------------
    # 이미지별로 어노테이션을 YOLO 형식으로 변환하고,
    # 생성된 라벨 파일과 이미지를 train/val 폴더로 복사
    for img_id in tqdm(image_ids_to_split, desc="(Stage 2/5) 학습/검증 데이터 YOLO 변환 및 분리"):
        # 이미지 메타 정보 조회 (파일명, 너비, 높이)
        img_info = image_info_map.get(img_id)
        if not img_info:
            continue

        file_name = img_info.get('file_name')
        img_width = img_info.get('width')
        img_height = img_info.get('height')

        # 필수 정보가 없으면 건너뜀
        if not (file_name and img_width and img_height):
            continue

        src_image_path = os.path.join(image_base_dir, file_name)
        # 이미지 파일이 존재하는지 확인
        if not os.path.exists(src_image_path):
            continue

        yolo_lines = []
        image_annotations = image_annotations_map.get(img_id, [])

        for anno in image_annotations:
            # 바운딩 박스 좌표와 카테고리 추출
            bbox = anno.get('bbox')
            category_id = anno.get('category_id')

            if bbox is None or category_id is None:
                # 잘못된 어노테이션은 무시
                continue

            x_min, y_min, bbox_width, bbox_height = bbox

            # 이미지 및 바운딩 박스 유효성 검사
            if img_width <= 0 or img_height <= 0 or bbox_width <= 0 or bbox_height <= 0:
                continue

            # COCO 바운딩 값을 YOLO(center x, center y, width, height) 정규화 형태로 변환
            x_center = (x_min + bbox_width / 2) / img_width
            y_center = (y_min + bbox_height / 2) / img_height
            norm_width = bbox_width / img_width
            norm_height = bbox_height / img_height

            yolo_class_id = category_id_map.get(category_id)
            if yolo_class_id is None:
                # 카테고리 매핑이 없으면 무시
                continue

            # 좌표 값이 0~1 범위를 벗어나지 않도록 클리핑
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))

            # 너무 작은 박스는 무시
            if norm_width < 1e-6 or norm_height < 1e-6:
                continue

            yolo_lines.append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        # 이미지가 train 또는 val 중 어디에 속하는지 판단
        subset = 'train' if img_id in train_ids else 'val'

        if yolo_lines:
            # YOLO 라벨 파일 생성 경로 설정
            label_output_path = os.path.join(
                yolo_labels_dir, subset, os.path.splitext(file_name)[0] + '.txt'
            )
            # 라벨 파일 쓰기
            with open(label_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))

        # 이미지 파일을 해당 subset 폴더로 복사
        dest_image_path = os.path.join(yolo_images_dir, subset, file_name)
        shutil.copy(src_image_path, dest_image_path)

    # 생성된 모든 클래스 매핑 정보를 반환
    yolo_to_categoryid = {cat['yolo_id']: cat['id'] for cat in all_categories_info}
    return final_class_names, yolo_to_categoryid


# ==================== data.yaml 설정 파일 생성 함수 ====================
# data.yaml 파일을 생성해 학습/검증 경로 및 클래스 정보를 저장
# 파일이 없으면 디렉토리를 생성해 파일을 작성
# 생성 완료 메시지를 출력
# ====================================================================
def create_yolo_yaml(yaml_file_path, output_data_dir, class_names_list):
    data_for_yaml = {
        'train': 'images/train_aug',
        'val': 'images/val',
        'nc': len(class_names_list),
        'names': class_names_list
    }
    os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_for_yaml, f, sort_keys=False, allow_unicode=True, indent=2)

    print(f"(Stage 5/5) data.yaml 생성 완료 (경로: {yaml_file_path})")


# ==================== 전체 파이프라인 실행 (main 블록) ===================
# 스크립트 실행 시 COCO 변환, 테스트 이미지 복사, 클래스 매핑 저장, YAML 생성 수행
# 필요한 디렉토리와 파일이 이미 있으면 전처리 단계를 건너뜀
# 실행 흐름을 순서대로 출력
# =====================================================================
if __name__ == '__main__':
    print("[INFO] 데이터 로드 중")
    output_yolo_data_dir = output_dir
    yaml_file_path = os.path.join(output_yolo_data_dir, 'data.yaml')
    class_names_path = os.path.join(output_yolo_data_dir, 'class_names.json')
    
    required_subdirs_for_skip = [
        os.path.join(output_yolo_data_dir, 'images', 'train'),
        os.path.join(output_yolo_data_dir, 'images', 'val'),
        os.path.join(output_yolo_data_dir, 'labels', 'train'),
        os.path.join(output_yolo_data_dir, 'labels', 'val'),
    ]
    can_skip_preprocessing = all(os.path.exists(p) for p in required_subdirs_for_skip) and os.path.exists(class_names_path)

    converted_class_names = None

    if can_skip_preprocessing:
        print("[INFO] 이미 모든 전처리 파일 존재 (Stage 5/5) 단계 바로 진행")
        try:
            with open(class_names_path, 'r', encoding='utf-8') as f:
                converted_class_names = json.load(f)
        except Exception: 
            converted_class_names = None 
    else:
        # -------------------- 입력 경로 및 JSON 수집 --------------------
        kaggle_data_root = input_dir
        json_annotations_dir = os.path.join(kaggle_data_root, 'train_annotations')
        json_paths = glob(os.path.join(json_annotations_dir, '**', '*.json'), recursive=True)

        if not json_paths:
            exit()

        image_base_dir = os.path.join(kaggle_data_root, 'train_images')
        if not os.path.exists(image_base_dir):
            exit()

        converted_class_names, yolo_to_categoryid_map = merge_and_convert_coco_to_yolo(
            json_paths, image_base_dir, output_yolo_data_dir, val_split=0.2
        )

        if converted_class_names and yolo_to_categoryid_map:
            # -------------------- 테스트 이미지 복사 --------------------
            test_src_dir = os.path.join(kaggle_data_root, 'test_images')
            test_dest_dir = os.path.join(output_yolo_data_dir, 'images', 'test')
            os.makedirs(test_dest_dir, exist_ok=True)

            if os.path.exists(test_src_dir):
                test_files = [f for f in os.listdir(test_src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for file_name in tqdm(test_files, desc="(Stage 3/5) 테스트 이미지 복사"):
                    shutil.copy(os.path.join(test_src_dir, file_name), test_dest_dir)

            # -------------------- 클래스 매핑 및 이름 저장 --------------------
            mapping_save_path = os.path.join(output_yolo_data_dir, 'yolo_to_categoryid.json')
            with open(mapping_save_path, "w", encoding="utf-8") as f:
                json.dump(yolo_to_categoryid_map, f, ensure_ascii=False, indent=2)

            with open(class_names_path, "w", encoding="utf-8") as f:
                json.dump(converted_class_names, f, ensure_ascii=False, indent=2)
            print("(Stage 4/5) 클래스 정보 및 ID 매핑 저장")
        else:
            exit()

    if converted_class_names is not None:
        create_yolo_yaml(yaml_file_path, output_yolo_data_dir, converted_class_names)
    else:
        pass