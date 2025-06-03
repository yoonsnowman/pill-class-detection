import json
import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np # np.array를 사용하기 위해 추가
from collections import Counter # 클래스 카운팅을 위해 추가


# ---- pill-detect-ai 폴더 열어서 상대 경로로 진행 ---------------------
input_dir = 'data/yolo'
output_dir = 'data/yolo/pill_yolo_format'
# -----------------------------------------------------------------

def merge_and_convert_coco_to_yolo(json_paths, image_base_dir, output_base_dir, val_split=0.2):
    """
    여러 개의 COCO 형식 JSON 어노테이션 파일을 병합하고, YOLO 형식으로 변환하며,
    학습 및 검증 세트로 데이터를 분할합니다.

    Args:
        json_paths (list): COCO JSON 어노테이션 파일 경로 리스트 (예: ['path/to/part1.json', 'path/to/part2.json']).
        image_base_dir (str): 원본 이미지 파일들이 있는 디렉토리의 베이스 경로.
                               (예: '/content/ai02-level1-project/train_images/')
        output_base_dir (str): 변환된 YOLO 형식 데이터가 저장될 최상위 디렉토리.
                               (예: '/content/datasets/pill_yolo_format')
        val_split (float): 검증 세트로 분할할 비율 (0.0 ~ 1.0).
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
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 카테고리 병합 및 YOLO class_id 매핑
            # .get()을 사용하여 키가 없을 때 오류 방지
            if 'categories' in data and data['categories']: # 'categories' 키가 있고 비어있지 않은지 확인
                for category in data['categories']:
                    if category.get('id') is not None and category['id'] not in category_id_map:
                        category_id_map[category['id']] = next_yolo_class_id
                        all_categories.append({'id': category['id'], 'name': category.get('name', f"class_{category['id']}"), 'yolo_id': next_yolo_class_id})
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
            print(f"Error: Could not decode JSON from {json_path}. Skipping.")
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
        # image_id가 문자열인 경우를 대비하여 int로 변환 시도
        try:
            img_id = int(img['id'])
        except (ValueError, TypeError):
            print(f"Warning: Non-integer image ID found: {img.get('id')}. Skipping image {img.get('file_name')}.")
            continue
        # .get()을 사용하여 키가 없을 때 오류 방지
        image_info_map[img_id] = {'file_name': img.get('file_name'), 'width': img.get('width'), 'height': img.get('height')}

    print(f"Number of images in image_info_map: {len(image_info_map)}")

    # 각 이미지의 모든 어노테이션을 묶기
    image_annotations_map = {}
    for anno in all_annotations:
        image_id_raw = anno.get('image_id')
        try:
            image_id = int(image_id_raw) # 어노테이션의 image_id도 int로 변환 시도
        except (ValueError, TypeError):
            print(f"Warning: Non-integer image_id found in annotation: {image_id_raw}. Skipping annotation {anno}.")
            continue

        if image_id is not None:
            if image_id not in image_annotations_map:
                image_annotations_map[image_id] = []
            image_annotations_map[image_id].append(anno)
        else:
            print(f"Warning: Annotation without 'image_id' found: {anno}. Skipping.")

    print(f"Number of images with annotations in image_annotations_map: {len(image_annotations_map)}")

    # 이미지 ID 리스트를 학습/검증 세트로 분할
    image_ids_to_split = list(image_annotations_map.keys())

    # *** IMPORTANT CHECK FOR EMPTY DATA ***
    if not image_ids_to_split:
        print("\nError: No images with annotations found after processing JSON files. Cannot perform train/val split.")
        print(f"Total annotations collected: {len(all_annotations)}")
        print(f"Content of all_annotations (first 5): {all_annotations[:min(5, len(all_annotations))]}")
        print(f"Content of image_annotations_map: {image_annotations_map}")
        print("Please verify that:")
        print("1. JSON files exist and are valid.")
        print("2. JSON files contain 'images' and 'annotations' sections.")
        print("3. 'image_id' in 'annotations' matches 'id' in 'images'.")
        print("4. All image IDs have at least one valid annotation.")
        return [] # Return empty list for class names and stop processing


    # --- MODIFIED STRATIFY LOGIC ---
    # stratify를 위한 라벨 생성
    # 각 이미지에 첫 번째 어노테이션의 클래스 ID를 사용합니다.
    # 이미지에 어노테이션이 없는 경우, 또는 유효한 category_id가 없는 경우는 None으로 처리합니다.
    stratify_labels_raw = []
    for img_id in image_ids_to_split:
        annotations_for_img = image_annotations_map.get(img_id)
        if annotations_for_img and annotations_for_img[0].get('category_id') is not None:
            coco_cat_id = annotations_for_img[0]['category_id']
            stratify_labels_raw.append(category_id_map.get(coco_cat_id))
        else:
            stratify_labels_raw.append(None) # 어노테이션 없거나 유효 ID 없는 경우

    # None 값 제거 및 유효한 클래스 ID만 필터링하여 stratify 대상 리스트 생성
    valid_stratify_labels = [label for label in stratify_labels_raw if label is not None]

    stratify_possible = True
    if len(set(valid_stratify_labels)) < 2: # 유효한 클래스가 단 하나이거나 전혀 없는 경우
        stratify_possible = False
        print("Warning: Only one class or no valid classes found for stratification. Performing simple train/val split.")
    else:
        # 각 클래스별 샘플 개수를 세어 1개인 클래스가 있는지 확인
        class_counts = Counter(valid_stratify_labels)
        for class_id, count in class_counts.items():
            if count < 2:
                stratify_possible = False
                print(f"Warning: Class ID {class_id} has only {count} member(s). Stratification for this class is not possible.")
                print("Performing simple train/val split for all data.")
                break # 단 하나라도 문제가 되는 클래스가 있다면 전체를 비-층화 분할

    if stratify_possible:
        # train_test_split의 stratify 인자에는 numpy 배열이 더 안정적입니다.
        train_ids, val_ids = train_test_split(
            np.array(image_ids_to_split),
            test_size=val_split,
            random_state=42,
            stratify=np.array(valid_stratify_labels) # 유효한 stratify_labels만 전달
        )
    else:
        # Fallback to non-stratified split if stratification is not possible
        train_ids, val_ids = train_test_split(
            np.array(image_ids_to_split),
            test_size=val_split,
            random_state=42
        )

    # train_test_split 결과가 numpy 배열이므로 다시 리스트로 변환
    train_ids = train_ids.tolist()
    val_ids = val_ids.tolist()
    # --- END OF MODIFIED STRATIFY LOGIC ---

    print(f"Total annotated images selected for split: {len(image_ids_to_split)}")
    print(f"Train images count: {len(train_ids)}")
    print(f"Validation images count: {len(val_ids)}")

    # 2. 데이터 변환 및 파일 복사
    print("Converting to YOLO format and copying files...")
    processed_images_count = 0
    for img_id in image_ids_to_split: # Only iterate over images that have annotations
        img_info = image_info_map.get(img_id)

        # In this updated logic, img_id in image_ids_to_split implies img_info should exist
        # and it should have valid file_name, width, height. But keeping checks for robustness.
        if not img_info:
            print(f"Warning: Image info not found for ID {img_id} (should not happen if image_ids_to_split is correctly built). Skipping.")
            continue

        file_name = img_info.get('file_name')
        img_width = img_info.get('width')
        img_height = img_info.get('height')

        if file_name is None or img_width is None or img_height is None:
            print(f"Warning: Incomplete image info for ID {img_id}. Skipping processing for this image.")
            continue

        # 원본 이미지 파일의 절대 경로를 만듭니다.
        src_image_path = os.path.join(image_base_dir, file_name)
        if not os.path.exists(src_image_path):
            print(f"Error: Image file not found at {src_image_path}. Skipping processing for {file_name}.")
            continue

        # 라벨 파일 내용 생성
        yolo_lines = []
        image_annotations = image_annotations_map.get(img_id, []) # Ensure it's a list even if empty

        if not image_annotations:
            # 이 경우는 image_ids_to_split에 포함된 이미지 ID는 아노테이션이 있어야 하므로,
            # 거의 발생하지 않아야 합니다. 하지만 혹시 모를 경우를 대비.
            print(f"Warning: No valid annotations found for image ID {img_id} after map creation. Skipping label creation.")
            # Skip creating label file, but still copy image if needed.
            # For YOLO, if no bbox, no label file is generated.

        for anno in image_annotations:
            bbox = anno.get('bbox')
            category_id = anno.get('category_id')

            if bbox is None or category_id is None:
                print(f"Warning: Annotation missing 'bbox' or 'category_id': {anno}. Skipping this specific annotation.")
                continue

            x_min, y_min, bbox_width, bbox_height = bbox

            # Ensure division by zero is avoided and dimensions are positive
            if img_width <= 0 or img_height <= 0:
                print(f"Warning: Invalid image dimensions for image ID {img_id} ({img_width}x{img_height}). Cannot normalize bbox. Skipping this image.")
                yolo_lines = [] # Clear any lines if image dimensions are bad
                break # Exit inner loop for this image
            if bbox_width < 0 or bbox_height < 0:
                 print(f"Warning: Invalid bbox dimensions for image ID {img_id} ({bbox_width}x{bbox_height}). Skipping this specific annotation.")
                 continue

            x_center = (x_min + bbox_width / 2) / img_width
            y_center = (y_min + bbox_height / 2) / img_height
            norm_width = bbox_width / img_width
            norm_height = bbox_height / img_height

            # Ensure category_id exists in the map
            yolo_class_id = category_id_map.get(category_id)
            if yolo_class_id is None:
                print(f"Warning: Category ID {category_id} not found in category_id_map for annotation {anno}. Skipping this specific annotation.")
                continue

            # Ensure normalized values are within [0, 1] (due to potential floating point errors or invalid bbox)
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))

            # Filter out annotations with zero or extremely small normalized dimensions
            if norm_width < 1e-6 or norm_height < 1e-6:
                print(f"Warning: Very small or zero dimension bbox for image {file_name}: {bbox}. Skipping annotation.")
                continue


            yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        # 학습/검증 세트 폴더 결정
        if img_id in train_ids:
            subset = 'train'
        elif img_id in val_ids:
            subset = 'val'
        else:
            # 이 경우는 image_ids_to_split에 포함된 모든 ID가 train_ids 또는 val_ids에 포함되어야 하므로 발생하지 않아야 함
            print(f"Warning: Image ID {img_id} not assigned to train/val splits unexpectedly. Skipping.")
            continue

        # YOLO 라벨 파일 저장: 유효한 yolo_lines가 있을 때만 파일 생성
        label_output_path = os.path.join(yolo_labels_dir, subset, os.path.splitext(file_name)[0] + '.txt')
        if yolo_lines:
            with open(label_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            
            processed_images_count += 1
        else:
            print(f"Info: No valid annotations for image {file_name} after filtering. Skipping label file creation.")


        # 이미지 파일 복사 (라벨 파일 생성 여부와 무관하게 이미지 자체는 복사)
        dest_image_path = os.path.join(yolo_images_dir, subset, file_name)
        shutil.copy(src_image_path, dest_image_path) # shutil.copy 사용 권장


    print(f"COCO to YOLO conversion complete! Successfully processed {processed_images_count} images with labels.")
        # YOLO class index → 실제 category_id(dl_idx) 매핑 dict 생성
    yolo_to_categoryid = {cat['yolo_id']: cat['id'] for cat in all_categories}

    # json 파일로 저장
    mapping_save_path = os.path.join(output_base_dir, 'yolo_to_categoryid.json')
    with open(mapping_save_path, "w", encoding="utf-8") as f:
        json.dump(yolo_to_categoryid, f, ensure_ascii=False, indent=2)
    print(f"YOLO→category_id 매핑 저장 완료: {mapping_save_path}")

    return final_class_names # 최종 클래스 이름 리스트 반환

# --- 코랩에서 실행 예시 ---
if __name__ == '__main__':
    # 0. Kaggle 데이터셋 다운로드 및 압축 해제 (이전 단계에서 완료했다고 가정)
    #    예시:
    #    !pip install kaggle
    #    from google.colab import files
    #    files.upload() # kaggle.json 업로드
    #    !mkdir -p ~/.kaggle
    #    !mv kaggle.json ~/.kaggle/
    #    !chmod 600 ~/.kaggle/kaggle.json
    #    !kaggle competitions download -c ai02-level1-project
    #    !unzip -qq ai02-level1-project.zip

    # 1. 파일 경로 설정
    # 캐글 데이터셋이 /content/ai02-level1-project/ 에 압축 해제되었다고 가정
    kaggle_data_root = input_dir

    # train_annotations 폴더 내의 모든 JSON 파일 경로를 재귀적으로 찾습니다.
    # 사진에서 본 경로: train_annotations -> K-001900... -> K-001900 -> *.json
    # 따라서 **/*.json 패턴을 사용하여 모든 하위 디렉토리를 탐색합니다.
    json_annotations_dir = os.path.join(kaggle_data_root, 'train_annotations')
    print(f"Listing contents of annotation directory: {json_annotations_dir}")
    #!ls -R {json_annotations_dir} # -R 옵션으로 하위 디렉토리까지 모두 출력 

    json_paths = glob(os.path.join(json_annotations_dir, '**', '*.json'), recursive=True)

    if not json_paths:
        print(f"Error: No JSON files found in {json_annotations_dir} or its subdirectories.")
        print("Please double check the path and make sure JSON files are there.")
        # 이 시점에서 JSON 파일을 못 찾으면 데이터 전처리를 진행할 수 없으므로, 스크립트 종료
        exit()
    else:
        print(f"Found {len(json_paths)} JSON files: {json_paths[:min(5, len(json_paths))]}...")
        if len(json_paths) > 5:
            print(f"(Showing first 5 paths only. Total: {len(json_paths)})")

    # 원본 이미지가 있는 폴더
    image_base_dir = os.path.join(kaggle_data_root, 'train_images')
    if not os.path.exists(image_base_dir):
        print(f"Error: train_images directory not found at {image_base_dir}. Please check your dataset extraction.")
        exit()

    output_yolo_data_dir = output_dir

    # 변환 스크립트 실행
    converted_class_names = merge_and_convert_coco_to_yolo(json_paths, image_base_dir, output_yolo_data_dir, val_split=0.2)

    # 변환이 성공적으로 완료되었는지 확인
    if converted_class_names: # converted_class_names가 비어있지 않다면 성공
        print("\n--- Data Conversion Successful! Proceeding with file operations. ---")

        # 2. 테스트 이미지 복사 (선택 사항이지만, 최종 추론을 위해 필요)
        print("\nCopying test images...")
        test_src_dir = os.path.join(kaggle_data_root, 'test_images')
        test_dest_dir = os.path.join(output_yolo_data_dir, 'images', 'test')
        os.makedirs(test_dest_dir, exist_ok=True)

        if os.path.exists(test_src_dir):
            copied_test_count = 0
            for file_name in os.listdir(test_src_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(os.path.join(test_src_dir, file_name), test_dest_dir)
                    copied_test_count += 1
            print(f"Copied {copied_test_count} test images to {test_dest_dir}!")
        else:
            print(f"Warning: Test images directory not found at {test_src_dir}. Skipping test image copy.")

        # 3. yolov5/data/pill.yaml 파일 내용 생성/업데이트
        print("\n--- YOLOv5 pill.yaml content ---")
        print(f"train: {output_yolo_data_dir}/images/train/")
        print(f"val: {output_yolo_data_dir}/images/val/")
        # 테스트 데이터셋을 YOLOv5의 검증 단계에서 포함하지 않는 경우, test: 라인은 필요 없습니다.
        # 추론 시에는 detect.py의 --source 옵션으로 test_images 폴더를 직접 지정합니다.
        # print(f"test: {output_yolo_data_dir}/images/test/")
        print(f"nc: {len(converted_class_names)}")
        print(f"names: {converted_class_names}")
        print("---------------------------------")

        # 최종적으로 데이터셋 폴더 구조 확인
        print("\nFinal dataset structure (first 5 files in each dir, and total counts):")

        def print_dir_contents(path, label):
            if os.path.exists(path):
                files = os.listdir(path)
                print(f"{label} ({len(files)} files): {files[:min(5, len(files))]}...")
            else:
                print(f"{label} directory not found: {path}")

        print_dir_contents(os.path.join(output_yolo_data_dir, 'images', 'train'), "Images Train")
        print_dir_contents(os.path.join(output_yolo_data_dir, 'labels', 'train'), "Labels Train")
        print_dir_contents(os.path.join(output_yolo_data_dir, 'images', 'val'), "Images Val")
        print_dir_contents(os.path.join(output_yolo_data_dir, 'labels', 'val'), "Labels Val")
        print_dir_contents(os.path.join(output_yolo_data_dir, 'images', 'test'), "Images Test")

    else:
        print("\n--- Data Conversion FAILED. Please review the errors and warnings above. ---")

