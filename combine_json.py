import os
import json
from tqdm import tqdm

def merge_coco_jsons_for_deep_structure(annotation_root_dir, image_dir, output_json_path, category_mapping=None):
    """
    제공된 깊은 폴더 구조(예: 'K-XXXX_json/K-YYYY/K-ZZZZ.json')에 맞춰
    개별 COCO 형식 JSON 파일들을 하나의 COCO 형식 JSON 파일로 병합합니다.
    각 개별 JSON 파일은 'images', 'annotations', 'categories' 키를 모두 포함한다고 가정합니다.

    Args:
        annotation_root_dir (str): 개별 주석 폴더들이 들어있는 최상위 디렉토리.
                                  (예: 'data/train_annotations_split' 또는 'data/val_annotations')
        image_dir (str): 해당 주석에 연결될 이미지 파일들이 들어있는 디렉토리.
                         (예: 'data/train_images_split' 또는 'data/val_images')
        output_json_path (str): 병합된 COCO JSON 파일이 저장될 경로 및 파일명.
                                (예: 'data/train_coco.json' 또는 'data/val_coco.json')
        category_mapping (dict, optional): 필요시 카테고리 ID를 재매핑하기 위한 딕셔너리.
                                           기본값은 None (기존 카테고리 ID 유지).
    """
    print(f"\n✨ {annotation_root_dir}의 JSON 파일과 {image_dir}의 이미지를 병합하여 {output_json_path} 생성 시작...")

    coco_format = {
        "info": {
            "description": "Merged Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "YourName",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 이미지 ID, 어노테이션 ID, 카테고리 ID는 병합 과정에서 재할당되어야 고유성을 유지할 수 있습니다.
    global_image_id_counter = 0
    global_annotation_id_counter = 0
    global_category_id_map = {} # {original_id: new_global_id}
    global_category_name_to_id = {} # {name: new_global_id}
    
    # 이미지 파일명과 매핑된 새 이미지 ID를 저장 (중복 방지 및 빠른 조회)
    image_filename_to_new_id = {} 

    # 1. image_dir의 모든 이미지 파일명 사전 스캔 (존재 여부 확인 및 COCO ID 할당)
    print(f"이미지 정보 사전 수집 중 ({image_dir})...")
    actual_image_filenames_in_dir = set()
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            actual_image_filenames_in_dir.add(filename)
    
    if not actual_image_filenames_in_dir:
        print(f"❌ 오류: '{image_dir}' 디렉토리에서 유효한 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return
        
    print(f"   ➡️ 총 {len(actual_image_filenames_in_dir)}개의 이미지 파일명 확인 완료.")
    print(f"   ➡️ 첫 5개 이미지 파일명 예시: {list(actual_image_filenames_in_dir)[:5]}")

    # 2. annotation_root_dir의 모든 주석 폴더를 탐색하며 JSON 파일 찾기
    print(f"'{annotation_root_dir}'에서 주석 파일 탐색 중...")
    
    # os.walk를 사용하여 모든 하위 디렉토리를 탐색합니다.
    all_json_files_found = []
    for dirpath, dirnames, filenames in os.walk(annotation_root_dir):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                all_json_files_found.append(os.path.join(dirpath, filename))

    if not all_json_files_found:
        print(f"⚠️ 경고: '{annotation_root_dir}' 및 하위 디렉토리에서 JSON 파일을 찾을 수 없습니다. 경로 또는 파일명을 확인해주세요.")
        return

    processed_annotation_files_count = 0
    
    # TQDM 프로그레스 바 적용
    for json_file_path in tqdm(all_json_files_found, desc="개별 주석 파일 처리 중"):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            # --- 이미지 정보 처리 (각 개별 JSON 파일에서 가져옴) ---
            if "images" in annotation_data and annotation_data["images"]:
                # 개별 JSON 파일에는 이미지가 하나만 있을 것으로 가정합니다.
                original_image_info = annotation_data["images"][0]
                original_file_name = original_image_info.get("file_name")
                
                if not original_file_name:
                    # print(f"   ⚠️ 경고: '{json_file_path}'의 이미지 정보에 'file_name'이 없습니다. 건너뜁니다.")
                    continue

                # 해당 이미지가 실제로 image_dir에 존재하는지 확인
                if original_file_name not in actual_image_filenames_in_dir:
                    # print(f"   ⚠️ 경고: 이미지 '{original_file_name}'가 '{image_dir}'에 없습니다. 주석 파일 '{json_file_path}'을 건너뜁니다.")
                    continue

                # 이미 처리된 이미지인지 확인 (파일명을 기준으로)
                if original_file_name not in image_filename_to_new_id:
                    # 새 이미지 ID 할당
                    new_image_id = global_image_id_counter
                    image_filename_to_new_id[original_file_name] = new_image_id
                    global_image_id_counter += 1

                    # COCO images 리스트에 추가
                    coco_format["images"].append({
                        "id": new_image_id,
                        "width": original_image_info.get("width", 0),
                        "height": original_image_info.get("height", 0),
                        "file_name": original_file_name,
                        "license": original_image_info.get("license", 0),
                        "flickr_url": original_image_info.get("flickr_url", ""),
                        "coco_url": original_image_info.get("coco_url", ""),
                        "date_captured": original_image_info.get("date_captured", "")
                    })
                else:
                    new_image_id = image_filename_to_new_id[original_file_name]
            else:
                # print(f"   ⚠️ 경고: '{json_file_path}'에 'images' 키가 없거나 비어있습니다. 건너뜁니다.")
                continue

            # --- 카테고리 정보 처리 ---
            if "categories" in annotation_data and annotation_data["categories"]:
                for cat in annotation_data["categories"]:
                    original_cat_id = cat.get("id")
                    cat_name = cat.get("name")
                    
                    if original_cat_id is None or cat_name is None:
                        # print(f"   ⚠️ 경고: '{json_file_path}'의 카테고리 정보에 'id' 또는 'name'이 없습니다. 건너뜁니다.")
                        continue

                    # 카테고리 이름으로 고유 ID를 관리
                    if cat_name not in global_category_name_to_id:
                        # 새로운 카테고리 ID 할당 (1부터 시작, 기존 ID와 충돌 방지)
                        new_global_cat_id = len(global_category_name_to_id) + 1 
                        global_category_name_to_id[cat_name] = new_global_cat_id
                        global_category_id_map[original_cat_id] = new_global_cat_id # 원본 ID -> 새 ID 매핑
                        
                        coco_format["categories"].append({
                            "id": new_global_cat_id,
                            "name": cat_name,
                            "supercategory": cat.get("supercategory", "")
                        })
                    else:
                        # 이미 추가된 카테고리라면 기존 ID 사용
                        global_category_id_map[original_cat_id] = global_category_name_to_id[cat_name]
            # else:
                # print(f"   ⚠️ 경고: '{json_file_path}'에 'categories' 키가 없거나 비어있습니다. 주석 처리 시 문제가 발생할 수 있습니다.")
            
            # --- 어노테이션 정보 처리 ---
            if "annotations" in annotation_data and annotation_data["annotations"]:
                for ann in annotation_data["annotations"]:
                    ann_image_id = new_image_id # 새로 할당된 이미지 ID 사용
                    
                    # 기존 category_id를 새롭게 매핑된 ID로 변환
                    ann_category_id = global_category_id_map.get(ann.get("category_id"), ann.get("category_id"))
                    
                    if ann_category_id is None:
                        # print(f"   ⚠️ 경고: '{json_file_path}'의 어노테이션에 'category_id'가 없거나 매핑할 수 없습니다. 건너뜁니다.")
                        continue

                    coco_format["annotations"].append({
                        "id": global_annotation_id_counter,
                        "image_id": ann_image_id,
                        "category_id": ann_category_id,
                        "bbox": ann.get("bbox", []), # bbox가 없을 경우 빈 리스트
                        "area": ann.get("area", 0),  # area가 없을 경우 0
                        "iscrowd": ann.get("iscrowd", 0),
                        "segmentation": ann.get("segmentation", ann.get("segmentation_data", []))
                    })
                    global_annotation_id_counter += 1
            # else:
                # print(f"   ⚠️ 경고: '{json_file_path}'에 'annotations' 키가 없거나 비어있습니다.")
            
            processed_annotation_files_count += 1

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {json_file_path} - {e}. 이 파일을 건너뜁니다.")
        except Exception as e:
            print(f"❌ 주석 처리 중 알 수 없는 오류 발생: {json_file_path} - {e}. 이 파일을 건너뜝니다.")
            import traceback
            traceback.print_exc()

    print(f"   ➡️ 총 {processed_annotation_files_count}개의 주석 파일이 성공적으로 처리되었습니다.")
    
    # 최종 카테고리 정렬 (ID 순으로)
    coco_format["categories"] = sorted(coco_format["categories"], key=lambda x: x["id"])

    # 병합된 JSON 파일 저장
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True) 
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, ensure_ascii=False, indent=4)
    
    print(f"✅ COCO JSON 파일 병합 완료: '{output_json_path}'")
    print(f"   - 총 이미지 수: {len(coco_format['images'])}")
    print(f"   - 총 주석 수: {len(coco_format['annotations'])}")
    print(f"   - 총 카테고리 수: {len(coco_format['categories'])}")

# --- 사용 예시 (이 부분을 실행합니다) ---
if __name__ == "__main__":
    # --- 1. 훈련 데이터셋 병합 ---
    print("=== 훈련 데이터셋 COCO JSON 병합 시작 ===")
    merge_coco_jsons_for_deep_structure(
        annotation_root_dir='data/train_annotations_split', # 훈련 이미지에 해당하는 주석이 모인 폴더
        image_dir='data/train_images_split',          # 훈련 이미지 폴더
        output_json_path='data/train_coco.json'
    )
    print("=== 훈련 데이터셋 COCO JSON 병합 완료 ===")

    # --- 2. 검증 데이터셋 병합 ---
    print("\n=== 검증 데이터셋 COCO JSON 병합 시작 ===")
    merge_coco_jsons_for_deep_structure(
        annotation_root_dir='data/val_annotations', # 검증 이미지에 해당하는 주석이 모인 폴더
        image_dir='data/val_images',                # 검증 이미지 폴더
        output_json_path='data/val_coco.json'
    )
    print("=== 검증 데이터셋 COCO JSON 병합 완료 ===")