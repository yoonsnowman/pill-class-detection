import json
import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
from collections import Counter
import argparse # 명령줄 인자 처리를 위해 추가

def merge_and_convert_coco_to_yolo(json_paths, image_base_dir, output_base_dir, val_split=0.2):
    """
    여러 개의 COCO 형식 JSON 어노테이션 파일을 병합하고, YOLO 형식으로 변환하며,
    학습 및 검증 세트로 데이터를 분할합니다.

    Args:
        json_paths (list): COCO JSON 어노테이션 파일 경로 리스트 (예: ['path/to/part1.json', 'path/to/part2.json']).
        image_base_dir (str): 원본 이미지 파일들이 있는 디렉토리의 베이스 경로.
        output_base_dir (str): 변환된 YOLO 형식 데이터가 저장될 최상위 디렉토리.
        val_split (float): 검증 세트로 분할할 비율 (0.0 ~ 1.0).
    Returns:
        list: 최종 YOLO 형식의 클래스 이름 리스트. 변환 실패 시 빈 리스트 반환.
    """

    # 0. 출력 디렉토리 구조 생성
    yolo_labels_dir = os.path.join(output_base_dir, 'labels')
    yolo_images_dir = os.path.join(output_base_dir, 'images')

    os.makedirs(os.path.join(yolo_labels_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_labels_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_images_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_images_dir, 'val'), exist_ok=True)

    # 1. 모든 JSON 파일 병합
    all_images = []
    all_annotations = []
    all_categories = []
    category_id_map = {} # COCO category_id를 YOLO class_id (0부터 시작)로 매핑
    next_yolo_class_id = 0

    print("Merging COCO JSON files...")
    for json_path in json_paths:
        print(f"Loading {json_path}...")
        try:
            # 인코딩 문제 해결을 위한 시도 순서: UTF-8 -> CP949 -> UTF-8 (errors='ignore')
            data = None
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                print(f"  Attempting to decode with 'utf-8' failed for {json_path}. Trying 'cp949'...")
                with open(json_path, 'r', encoding='cp949') as f:
                    data = json.load(f)
            except Exception as e: # 다른 종류의 파일 열기 오류
                print(f"  An unexpected error occurred while opening {json_path} with standard encodings: {e}. Trying 'utf-8' with 'errors=ignore' (data loss possible!)...")
                with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)

            if data is None: # 데이터가 로드되지 않은 경우 (예: 상위 try-except 블록에서 이미 처리된 경우)
                continue

            # 카테고리 병합 및 YOLO class_id 매핑
            if 'categories' in data and data['categories']:
                for category in data['categories']:
                    coco_cat_id = category.get('id')
                    if coco_cat_id is not None and coco_cat_id not in category_id_map:
                        category_id_map[coco_cat_id] = next_yolo_class_id
                        all_categories.append({'id': coco_cat_id, 'name': category.get('name', f"class_{coco_cat_id}"), 'yolo_id': next_yolo_class_id})
                        next_yolo_class_id += 1
            else:
                print(f"Warning: No valid 'categories' found in {json_path}. Skipping categories from this file.")

            if 'images' in data:
                all_images.extend(data['images'])
            else:
                print(f"Warning: No 'images' found in {json_path}. Skipping images from this file.")

            if 'annotations' in data:
                all_annotations.extend(data['annotations'])
            else:
                print(f"Warning: No 'annotations' found in {json_path}. Skipping annotations from this file.")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON (invalid format) from {json_path}. Skipping.")
        except KeyError as e:
            print(f"Error: Missing expected key {e} in {json_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {json_path}: {e}. Skipping.")


    # 최종 클래스 이름 리스트 (YOLO class_id 순서대로)
    final_class_names = [cat['name'] for cat in sorted(all_categories, key=lambda x: x['yolo_id'])]
    num_classes = len(final_class_names)

    print(f"Merged total {len(all_images)} images and {len(all_annotations)} annotations.")
    print(f"Detected {num_classes} unique classes.")
    print(f"Final Class Names (YOLO Order): {final_class_names}")

    # 이미지 정보를 image_id를 키로 하는 딕셔너리로 저장하여 빠르게 접근
    image_info_map = {}
    for img in all_images:
        try:
            img_id = int(img['id'])
        except (ValueError, TypeError):
            print(f"Warning: Non-integer image ID found: {img.get('id')}. Skipping image {img.get('file_name')}.")
            continue
        image_info_map[img_id] = {'file_name': img.get('file_name'), 'width': img.get('width'), 'height': img.get('height')}

    print(f"Number of images in image_info_map: {len(image_info_map)}")

    # 각 이미지의 모든 어노테이션을 묶기
    image_annotations_map = {}
    for anno in all_annotations:
        image_id_raw = anno.get('image_id')
        try:
            image_id = int(image_id_raw)
        except (ValueError, TypeError):
            print(f"Warning: Non-integer image_id found in annotation: {image_id_raw}. Skipping annotation {anno}.")
            continue

        if image_id is not None:
            if image_id in image_info_map:
                if image_id not in image_annotations_map:
                    image_annotations_map[image_id] = []
                image_annotations_map[image_id].append(anno)
            else:
                print(f"Warning: Annotation with image_id {image_id} refers to an image not found in image_info_map. Skipping annotation.")
        else:
            print(f"Warning: Annotation without 'image_id' found: {anno}. Skipping.")

    print(f"Number of images with annotations in image_annotations_map: {len(image_annotations_map)}")

    # 이미지 ID 리스트를 학습/검증 세트로 분할
    image_ids_to_split = list(image_annotations_map.keys())

    if not image_ids_to_split:
        print("\nError: No images with annotations found after processing JSON files. Cannot perform train/val split.")
        print(f"Total annotations collected: {len(all_annotations)}")
        print("Please verify that:")
        print("1. JSON files exist and are valid.")
        print("2. JSON files contain 'images' and 'annotations' sections.")
        print("3. 'image_id' in 'annotations' matches 'id' in 'images'.")
        print("4. All image IDs have at least one valid annotation.")
        return []

    # --- STRATIFY LOGIC ---
    stratify_labels_raw = []
    for img_id in image_ids_to_split:
        annotations_for_img = image_annotations_map.get(img_id)
        if annotations_for_img and annotations_for_img[0].get('category_id') is not None:
            coco_cat_id = annotations_for_img[0]['category_id']
            if coco_cat_id in category_id_map:
                stratify_labels_raw.append(category_id_map.get(coco_cat_id))
            else:
                stratify_labels_raw.append(None)
        else:
            stratify_labels_raw.append(None)

    valid_indices_for_stratify = [i for i, label in enumerate(stratify_labels_raw) if label is not None]
    stratify_image_ids = [image_ids_to_split[i] for i in valid_indices_for_stratify]
    stratify_labels = [stratify_labels_raw[i] for i in valid_indices_for_stratify]

    stratify_possible = True
    if len(set(stratify_labels)) < 2:
        stratify_possible = False
        print("Warning: Only one class or no valid classes found for stratification. Performing simple train/val split.")
    else:
        class_counts = Counter(stratify_labels)
        min_samples_per_class = int(1 / val_split) if val_split > 0 else 2
        for class_id, count in class_counts.items():
            if count < min_samples_per_class:
                stratify_possible = False
                class_name_for_warning = final_class_names[class_id] if class_id < len(final_class_names) else 'Unknown'
                print(f"Warning: Class ID {class_id} (name: {class_name_for_warning}) has only {count} member(s), which is less than {min_samples_per_class} for stratification. Performing simple train/val split for all data.")
                break

    if stratify_possible:
        train_strat_ids, val_strat_ids = train_test_split(
            np.array(stratify_image_ids),
            test_size=val_split,
            random_state=42,
            stratify=np.array(stratify_labels)
        )
        non_stratified_image_ids = [img_id for img_id in image_ids_to_split if img_id not in stratify_image_ids]
        train_ids = train_strat_ids.tolist() + non_stratified_image_ids
        val_ids = val_strat_ids.tolist()
    else:
        train_ids, val_ids = train_test_split(
            np.array(image_ids_to_split),
            test_size=val_split,
            random_state=42
        )

    train_ids = train_ids.tolist()
    val_ids = val_ids.tolist()
    # --- END OF STRATIFY LOGIC ---

    print(f"Total annotated images selected for split: {len(image_ids_to_split)}")
    print(f"Train images count: {len(train_ids)}")
    print(f"Validation images count: {len(val_ids)}")

    # 2. 데이터 변환 및 파일 복사
    print("Converting to YOLO format and copying files...")
    processed_images_count = 0
    for img_id in image_ids_to_split:
        img_info = image_info_map.get(img_id)

        if not img_info:
            print(f"Warning: Image info not found for ID {img_id}. Skipping.")
            continue

        file_name = img_info.get('file_name')
        img_width = img_info.get('width')
        img_height = img_info.get('height')

        if file_name is None or img_width is None or img_height is None:
            print(f"Warning: Incomplete image info for ID {img_id}. Skipping processing for this image.")
            continue

        src_image_path = os.path.join(image_base_dir, file_name)
        if not os.path.exists(src_image_path):
            print(f"Error: Image file not found at {src_image_path}. Skipping processing for {file_name}.")
            continue

        yolo_lines = []
        image_annotations = image_annotations_map.get(img_id, [])

        if not image_annotations:
            print(f"Info: No annotations for image ID {img_id} or all annotations were invalid. Skipping label file creation.")

        for anno in image_annotations:
            bbox = anno.get('bbox')
            category_id = anno.get('category_id')

            if bbox is None or category_id is None:
                print(f"Warning: Annotation missing 'bbox' or 'category_id': {anno}. Skipping this specific annotation.")
                continue

            x_min, y_min, bbox_width, bbox_height = bbox

            if img_width <= 0 or img_height <= 0:
                print(f"Warning: Invalid image dimensions for image ID {img_id} ({img_width}x{img_height}). Cannot normalize bbox. Skipping all annotations for this image.")
                yolo_lines = []
                break
            if bbox_width < 0 or bbox_height < 0:
                print(f"Warning: Invalid bbox dimensions for image ID {img_id} ({bbox_width}x{bbox_height}). Skipping this specific annotation.")
                continue

            x_center = (x_min + bbox_width / 2) / img_width
            y_center = (y_min + bbox_height / 2) / img_height
            norm_width = bbox_width / img_width
            norm_height = bbox_height / img_height

            yolo_class_id = category_id_map.get(category_id)
            if yolo_class_id is None:
                print(f"Warning: Category ID {category_id} not found in category_id_map for annotation {anno}. Skipping this specific annotation.")
                continue

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))

            if norm_width < 1e-6 or norm_height < 1e-6:
                print(f"Warning: Very small or zero dimension bbox for image {file_name}: {bbox}. Skipping annotation.")
                continue

            yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        subset = None
        if img_id in train_ids:
            subset = 'train'
        elif img_id in val_ids:
            subset = 'val'
        else:
            print(f"Warning: Image ID {img_id} not assigned to train/val splits unexpectedly. Skipping.")
            continue
        
        label_output_path = os.path.join(yolo_labels_dir, subset, os.path.splitext(file_name)[0] + '.txt')
        if yolo_lines:
            with open(label_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            processed_images_count += 1
        else:
            print(f"Info: No valid annotations for image {file_name} after filtering. Skipping label file creation but copying image.")

        dest_image_path = os.path.join(yolo_images_dir, subset, file_name)
        try:
            shutil.copy(src_image_path, dest_image_path)
        except Exception as e:
            print(f"Error copying image {src_image_path} to {dest_image_path}: {e}. Skipping image copy.")

    print(f"COCO to YOLO conversion complete! Successfully processed {processed_images_count} images with labels.")
    return final_class_names

def create_yolo_data_yaml(output_dir, class_names):
    """
    YOLO 학습을 위한 data.yaml 파일을 생성합니다.
    """
    # Windows 경로를 /로 변환하여 yaml 경로에 적합하게 만듭니다.
    # 이전에 '1'이 붙은 것은 오류였으므로 제거합니다.
    output_dir_for_yaml = output_dir.replace(os.sep, '/')

    yaml_content = f"""
train: {output_dir_for_yaml}/images/train
val: {output_dir_for_yaml}/images/val
# test: {output_dir_for_yaml}/images/test # 테스트 이미지는 추론 시 detect.py --source로 지정

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(output_dir, 'data.yaml') # YOLOv5에서 일반적으로 사용하는 이름
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content.strip())
    print(f"\nCreated YOLO data.yaml file at: {yaml_path}")
    print(yaml_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO format and split into train/val sets.")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Path to the root directory of the Kaggle dataset (e.g., /path/to/ai02-level1-project/). This directory should contain 'train_images', 'train_annotations', 'test_images' (optional).")
    parser.add_argument('--output_dir', type=str, default='./datasets/pill_yolo_format',
                        help="Base directory where converted YOLO format data will be saved. Default is './datasets/pill_yolo_format'.")
    parser.add_argument('--val_split', type=float, default=0.2,
                        help="Validation set split ratio (0.0 to 1.0). Default is 0.2.")
    parser.add_argument('--copy_test_images', action='store_true',
                        help="Enable this flag to copy test images to the output directory.")

    args = parser.parse_args()

    # 1. 파일 경로 설정
    kaggle_data_root = args.data_root
    json_annotations_dir = os.path.join(kaggle_data_root, 'train_annotations')
    image_base_dir = os.path.join(kaggle_data_root, 'train_images')
    output_yolo_data_dir = args.output_dir

    print(f"Checking annotation directory: {json_annotations_dir}")
    json_paths = glob(os.path.join(json_annotations_dir, '**', '*.json'), recursive=True)
    json_paths = [os.path.normpath(p) for p in json_paths] # 경로 정규화

    if not json_paths:
        print(f"Error: No JSON files found in {json_annotations_dir} or its subdirectories.")
        print("Please double check the path and make sure JSON files are there.")
        exit(1)
    else:
        print(f"Found {len(json_paths)} JSON files: {json_paths[:min(5, len(json_paths))]}...")
        if len(json_paths) > 5:
            print(f"(Showing first 5 paths only. Total: {len(json_paths)})")

    if not os.path.exists(image_base_dir):
        print(f"Error: train_images directory not found at {image_base_dir}. Please check your dataset extraction.")
        exit(1)

    converted_class_names = merge_and_convert_coco_to_yolo(json_paths, image_base_dir, output_yolo_data_dir, val_split=args.val_split)

    if converted_class_names:
        print("\n--- Data Conversion Successful! ---")

        create_yolo_data_yaml(output_yolo_data_dir, converted_class_names)

        if args.copy_test_images:
            print("\nCopying test images...")
            test_src_dir = os.path.join(kaggle_data_root, 'test_images')
            test_dest_dir = os.path.join(output_yolo_data_dir, 'images', 'test')
            os.makedirs(test_dest_dir, exist_ok=True)

            if os.path.exists(test_src_dir):
                copied_test_count = 0
                for file_name in os.listdir(test_src_dir):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')) and not file_name.startswith('.'):
                        try:
                            shutil.copy(os.path.join(test_src_dir, file_name), test_dest_dir)
                            copied_test_count += 1
                        except Exception as e:
                            print(f"Error copying test image {file_name}: {e}")
                print(f"Copied {copied_test_count} test images to {test_dest_dir}!")
            else:
                print(f"Warning: Test images directory not found at {test_src_dir}. Skipping test image copy.")

        print("\nFinal dataset structure (first 5 files in each dir, and total counts):")

        def print_dir_contents(path, label):
            if os.path.exists(path):
                files = os.listdir(path)
                display_files = [f for f in files if not f.startswith('.')]
                print(f"{label} ({len(display_files)} files): {display_files[:min(5, len(display_files))]}...")
            else:
                print(f"{label} directory not found: {path}")

        print_dir_contents(os.path.join(output_yolo_data_dir, 'images', 'train'), "Images Train")
        print_dir_contents(os.path.join(output_yolo_data_dir, 'labels', 'train'), "Labels Train")
        print_dir_contents(os.path.join(output_yolo_data_dir, 'images', 'val'), "Images Val")
        print_dir_contents(os.path.join(output_yolo_data_dir, 'labels', 'val'), "Labels Val")
        if args.copy_test_images:
            print_dir_contents(os.path.join(output_yolo_data_dir, 'images', 'test'), "Images Test")

    else:
        print("\n--- Data Conversion FAILED. Please review the errors and warnings above. ---")
        exit(1)