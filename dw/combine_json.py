import os
import json
from tqdm import tqdm

def merge_coco_jsons_from_image_dir_based_filter(annotation_root_dir, image_dir, output_json_path):
    """
    image_dir에 존재하는 이미지들의 주석(JSON)만 기준으로 병합하는 함수
    """
    print(f"\n✨ '{image_dir}'의 이미지들만 기준으로 {annotation_root_dir}의 주석 파일을 병합합니다...")

    coco_format = {
        "info": {
            "description": "Merged Dataset (Filtered by image_dir)",
            "version": "1.0",
            "year": 2024,
            "contributor": "dw",
            "date_created": "2025-06-03"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    global_image_id_counter = 0
    global_annotation_id_counter = 0
    global_category_name_to_id = {}
    global_category_id_map = {}
    image_filename_to_new_id = {}

    # ✅ 기준이 되는 이미지들: image_dir에 실제 존재하는 파일들
    valid_image_filenames = {
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    }

    if not valid_image_filenames:
        print(f"❌ '{image_dir}'에 유효한 이미지가 없습니다.")
        return

    # ✅ 모든 JSON을 뒤지되, image_dir 기준 이미지에 해당하는 JSON만 병합
    json_files = []
    for dirpath, _, filenames in os.walk(annotation_root_dir):
        for fname in filenames:
            if fname.endswith('.json'):
                json_files.append(os.path.join(dirpath, fname))

    for json_path in tqdm(json_files, desc="주석 필터링 병합 중"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data.get('images'):
                continue

            image_info = data['images'][0]
            file_name = image_info.get('file_name')

            if file_name not in valid_image_filenames:
                continue  # ✅ image_dir 기준에 없으면 무시

            if file_name not in image_filename_to_new_id:
                new_id = global_image_id_counter
                image_filename_to_new_id[file_name] = new_id
                global_image_id_counter += 1

                coco_format["images"].append({
                    "id": new_id,
                    "width": image_info.get("width", 0),
                    "height": image_info.get("height", 0),
                    "file_name": file_name,
                    "license": image_info.get("license", 0),
                    "flickr_url": image_info.get("flickr_url", ""),
                    "coco_url": image_info.get("coco_url", ""),
                    "date_captured": image_info.get("date_captured", "")
                })

            new_image_id = image_filename_to_new_id[file_name]

            for cat in data.get("categories", []):
                cat_name = cat.get("name")
                if not cat_name:
                    continue
                if cat_name not in global_category_name_to_id:
                    new_cat_id = len(global_category_name_to_id) + 1
                    global_category_name_to_id[cat_name] = new_cat_id
                    global_category_id_map[cat["id"]] = new_cat_id
                    coco_format["categories"].append({
                        "id": new_cat_id,
                        "name": cat_name,
                        "supercategory": cat.get("supercategory", "")
                    })
                else:
                    global_category_id_map[cat["id"]] = global_category_name_to_id[cat_name]

            for ann in data.get("annotations", []):
                coco_format["annotations"].append({
                    "id": global_annotation_id_counter,
                    "image_id": new_image_id,
                    "category_id": global_category_id_map.get(ann["category_id"], ann["category_id"]),
                    "bbox": ann.get("bbox", []),
                    "area": ann.get("area", 0),
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": ann.get("segmentation", ann.get("segmentation_data", []))
                })
                global_annotation_id_counter += 1

        except Exception as e:
            print(f"❌ JSON 오류 무시: {json_path} - {e}")

    coco_format["categories"] = sorted(coco_format["categories"], key=lambda x: x["id"])

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, ensure_ascii=False, indent=4)

    print(f"✅ 저장 완료: {output_json_path}")
    print(f"   - 이미지 수: {len(coco_format['images'])}")
    print(f"   - 주석 수: {len(coco_format['annotations'])}")
    print(f"   - 카테고리 수: {len(coco_format['categories'])}")

# 예시 실행
if __name__ == "__main__":
    merge_coco_jsons_from_image_dir_based_filter(
        annotation_root_dir='data/detr/train_annotations',
        image_dir='data/detr/train_images',
        output_json_path='data/detr/annotations/train.json'
    )
