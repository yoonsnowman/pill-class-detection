import os
import shutil
from collections import Counter, defaultdict
from tqdm import tqdm
import cv2
import albumentations as A
import random
import argparse
import statistics
import configs.config_paths as cc

"""
🔧 사용법:

## 1. train 이미지 증강 폴더로 복사 (단, 폴더가 이미 생성되어 내용을 갖고 있으면 건너뜀)
python -m scripts.oversample --train_copy

## 2. 특정 수량 이하 클래스 출력
python -m scripts.oversample --list 50

## 3. 전체 클래스 -> 목표 수량까지 증강
python -m scripts.oversample --aug all --target 50

## 4. 특정 클래스 -> 목표 수량까지 증강
python -m scripts.oversample --aug 2 5 8 --target 50
"""

# ───────────────────────────────
# 📁 전역 경로 설정 (configs.config_paths.py 기반)
base_label_dir = cc.TRAIN_LB_DIR
base_image_dir = cc.TRAIN_IMG_DIR
norm_base_label_dir = os.path.normpath(base_label_dir)
norm_base_image_dir = os.path.normpath(base_image_dir)
aug_label_dir = os.path.join(
    os.path.dirname(norm_base_label_dir),
    os.path.basename(norm_base_label_dir) + '_aug'
)
aug_image_dir = os.path.join(
    os.path.dirname(norm_base_image_dir),
    os.path.basename(norm_base_image_dir) + '_aug'
)

# ───────────────────────────────
# 증강기 정의 (밝기&명도, 색조&채도&명도, 확대&축소)
augmentor = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=10,
        val_shift_limit=10,
        p=0.3
    ),
    A.Affine(
        scale=(0.9, 1.1),
        p=0.3
    )
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))

# ───────────────────────────────
# 원본 이미지 파일 찾기 함수
def find_original_image_file(base_name_of_image):
    """주어진 기본 이름으로 원본 이미지 폴더에서 이미지 파일을 찾습니다 (jpg, jpeg, png 지원)."""
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(base_image_dir, base_name_of_image + ext)
        if os.path.exists(path):
            return path
    return None

# ───────────────────────────────
# 메인 로직 함수
def main_logic(args):
    """스크립트의 주요 로직을 처리합니다."""

    # 증강 폴더 생성 (없으면 생성, 있어도 오류 없음)
    os.makedirs(aug_label_dir, exist_ok=True)
    os.makedirs(aug_image_dir, exist_ok=True)

    print("--- 경로 설정 ---")
    print(f"원본 라벨 폴더: {base_label_dir}")
    print(f"원본 이미지 폴더: {base_image_dir}")
    print(f"증강 라벨 폴더: {aug_label_dir}")
    print(f"증강 이미지 폴더: {aug_image_dir}")

    # ───────────────────────────────
    # (1) --train_copy 옵션이 주어졌을 때, 
    # 증강 폴더가 비어 있으면 복사, 아니라면 건너뜀
    if args.train_copy:
        # aug 폴더에 .txt 또는 이미지 파일이 하나라도 있으면 이미 복사된 것으로 간주
        existing_labels = [f for f in os.listdir(aug_label_dir) if f.endswith('.txt')]
        existing_images = [f for f in os.listdir(aug_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if existing_labels or existing_images:
            print(f"🛑 이미 파일이 존재하므로 (Stage 3/4) 라벨 수집 단계로 건너뜁니다.")
        else:
            for img_file in tqdm(os.listdir(base_image_dir), desc='(Stage 1/4) 원본 이미지 복사'):
                src_img_path = os.path.join(base_image_dir, img_file)
                dst_img_path = os.path.join(aug_image_dir, img_file)
                if os.path.isfile(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
            for lbl_file in tqdm(os.listdir(base_label_dir), desc='(Stage 2/4) 원본 라벨 복사'):
                src_lbl_path = os.path.join(base_label_dir, lbl_file)
                dst_lbl_path = os.path.join(aug_label_dir, lbl_file)
                if os.path.isfile(src_lbl_path):
                    shutil.copy(src_lbl_path, dst_lbl_path)
            print("✅ 원본 데이터 복사 완료.")

    class_counts = Counter()
    file_map = defaultdict(list)

    label_files_in_aug = [f for f in os.listdir(aug_label_dir) if f.endswith('.txt')]

    if not label_files_in_aug and not args.list:
        print(f"🛑 증강 라벨 폴더 '{aug_label_dir}'에 분석할 .txt 파일이 없습니다. 먼저 --train_copy를 실행하거나 라벨을 추가해주세요.")
        if not args.aug:
            return

    if label_files_in_aug:
        for file_name in tqdm(label_files_in_aug, desc='(Stage 3/4) 라벨 수집'):
            file_path = os.path.join(aug_label_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    classes_in_file = []
                    for line in lines:
                        stripped_line = line.strip()
                        if not stripped_line:
                            continue
                        parts = stripped_line.split()
                        if not parts:
                            continue
                        try:
                            cls_id = int(float(parts[0]))
                            class_counts[cls_id] += 1
                            classes_in_file.append(cls_id)
                        except (ValueError, IndexError):
                            print(f"⚠️ 파일 '{file_name}'의 라인 형식 오류 무시: {stripped_line}")
                    for unique_cls_id in set(classes_in_file):
                        file_map[unique_cls_id].append(file_name)
            except Exception as e:
                print(f"⚠️ 파일 '{file_name}' 처리 중 오류 발생: {e}")

    # ───────────────────────────────
    # "--list" 옵션 처리
    if args.list is not None:
        if not class_counts:
            print(f"\n🛑 '{aug_label_dir}' 폴더에 분석할 라벨 데이터가 없거나, 모든 라벨 파일이 비어있습니다.")
        else:
            print("\n📊 현재 증강 폴더의 클래스별 라벨 내 객체 수:")
            for cls_id_sorted in sorted(class_counts.keys()):
                print(f"Class {cls_id_sorted:2d}: {class_counts[cls_id_sorted]}개")
            values = list(class_counts.values())
            if values:
                mode_val = statistics.mode(values)
                median_val = int(statistics.median(values))
                rare_classes = sorted([cls for cls, count in class_counts.items() if count < median_val])
                print(f"\n📌 최빈값 (mode): {mode_val}개")
                print(f"📌 중앙값 (median): {median_val}개")
                if rare_classes:
                    print(f"📉 중앙값 미만 클래스: {', '.join(map(str, rare_classes))}")
                else:
                    print("📉 중앙값 미만 클래스가 없습니다.")
            else:
                print("\n🛑 통계 정보를 계산할 데이터가 충분하지 않습니다.")

        if isinstance(args.list, str) and args.list.isdigit():
            threshold = int(args.list)
            if class_counts and values:
                below_threshold = sorted([cls for cls, count in class_counts.items() if count < threshold])
                if below_threshold:
                    print(f"📉 {threshold}개 미만 클래스: {', '.join(map(str, below_threshold))}")
                else:
                    print(f"📉 {threshold}개 미만인 클래스가 없습니다.")
            else:
                print(f"\n🛑 {threshold}개 미만 클래스를 확인할 데이터가 없습니다.")

        # "--aug" 옵션도 같이 주어졌다면, 아래 증강 로직으로 계속 진행
        if not args.aug:
            return

    # ───────────────────────────────
    # "--aug" 옵션 처리
    if not args.aug:
        print("\n🛑 증강할 클래스가 지정되지 않았습니다. (--aug 옵션 사용)")
        print("   데이터 복사(--train_copy) 또는 목록 확인(--list)만 수행된 경우 정상입니다.")
        return

    if args.aug[0].lower() == 'all':
        target_classes_to_augment = sorted(class_counts.keys())
        if not target_classes_to_augment:
            print(f"🛑 현재 증강 폴더 '{aug_label_dir}'에 데이터가 없어 'all' 옵션으로 증강할 클래스가 없습니다.")
            return
    else:
        try:
            target_classes_to_augment = sorted(list(set(map(int, args.aug))))
        except ValueError:
            print("❌ 오류: --aug 로 전달된 클래스 번호는 정수여야 합니다. (예: --aug 0 1 2 또는 --aug all)")
            return

    target_per_class = args.target
    final_counts_after_aug = class_counts.copy()

    if target_classes_to_augment:
        print(f"\n(Stage 4/4) 소수 클래스 데이터 증강 시작")
        for cls_to_aug in target_classes_to_augment:
            current_class_obj_count = final_counts_after_aug.get(cls_to_aug, 0)
            if current_class_obj_count >= target_per_class:
                print(f"🟢 Class {cls_to_aug}: 이미 목표 수량({target_per_class}개) 이상({current_class_obj_count}개)입니다. 건너뜁니다.")
                continue

            num_needed_for_class = target_per_class - current_class_obj_count
            candidate_label_files_for_aug = file_map.get(cls_to_aug, [])
            if not candidate_label_files_for_aug:
                print(f"⚠️ Class {cls_to_aug}: 이 클래스를 포함하는 이미지가 '{aug_label_dir}'에 없습니다. 증강할 수 없습니다.")
                continue

            pbar = tqdm(
                total=num_needed_for_class,
                desc=f"🧬 Class {cls_to_aug} 증강 중 ({current_class_obj_count}/{target_per_class})"
            )
            augmented_count_for_this_class = 0
            attempt_count = 0
            max_attempts_per_needed = 10
            total_max_attempts = num_needed_for_class * max_attempts_per_needed

            while augmented_count_for_this_class < num_needed_for_class and attempt_count < total_max_attempts:
                attempt_count += 1
                source_label_filename_in_aug = random.choice(candidate_label_files_for_aug)
                base_name_of_source_in_aug = os.path.splitext(source_label_filename_in_aug)[0]

                # 원본 이미지는 항상 base_image_dir에서 가져옴
                source_image_path = find_original_image_file(base_name_of_source_in_aug)
                if not source_image_path:
                    continue

                try:
                    image_to_augment = cv2.imread(source_image_path)
                    if image_to_augment is None:
                        continue

                    bboxes_in_image, class_labels_in_image = [], []
                    source_label_file_path_in_aug = os.path.join(aug_label_dir, source_label_filename_in_aug)
                    with open(source_label_file_path_in_aug, 'r', encoding='utf-8') as f_label:
                        for line in f_label:
                            stripped_line = line.strip()
                            if not stripped_line:
                                continue
                            parts = stripped_line.split()
                            if not parts:
                                continue
                            try:
                                cls_id_in_label = int(float(parts[0]))
                                bbox_coords = list(map(float, parts[1:]))
                                if len(bbox_coords) == 4:
                                    bboxes_in_image.append(bbox_coords)
                                    class_labels_in_image.append(cls_id_in_label)
                            except (ValueError, IndexError):
                                pass

                    if not bboxes_in_image:
                        continue
                except Exception:
                    continue

                try:
                    augmented_data = augmentor(
                        image=image_to_augment,
                        bboxes=bboxes_in_image,
                        class_labels=class_labels_in_image
                    )
                    augmented_image = augmented_data['image']
                    augmented_bboxes = augmented_data['bboxes']
                    augmented_class_labels = augmented_data['class_labels']
                    if not augmented_bboxes:
                        continue
                except Exception:
                    continue

                new_file_base_name = f"{base_name_of_source_in_aug}_aug_{final_counts_after_aug.get(cls_to_aug, 0)}"
                original_img_extension = os.path.splitext(source_image_path)[1]
                output_augmented_image_path = os.path.join(aug_image_dir, new_file_base_name + original_img_extension)
                output_augmented_label_path = os.path.join(aug_label_dir, new_file_base_name + '.txt')

                _idx_collision = 0
                while os.path.exists(output_augmented_image_path) or os.path.exists(output_augmented_label_path):
                    _idx_collision += 1
                    new_file_base_name = f"{base_name_of_source_in_aug}_aug_{final_counts_after_aug.get(cls_to_aug, 0)}_{_idx_collision}"
                    output_augmented_image_path = os.path.join(aug_image_dir, new_file_base_name + original_img_extension)
                    output_augmented_label_path = os.path.join(aug_label_dir, new_file_base_name + '.txt')

                try:
                    cv2.imwrite(output_augmented_image_path, augmented_image)
                    with open(output_augmented_label_path, 'w', encoding='utf-8') as f_out_label:
                        obj_added_for_target_class = False
                        for aug_cls_id, aug_bbox in zip(augmented_class_labels, augmented_bboxes):
                            clamped_bbox = [max(0.0, min(1.0, coord)) for coord in aug_bbox]
                            f_out_label.write(f"{int(aug_cls_id)} {' '.join(f'{c:.6f}' for c in clamped_bbox)}\n")
                            final_counts_after_aug[int(aug_cls_id)] = final_counts_after_aug.get(int(aug_cls_id), 0) + 1
                            if int(aug_cls_id) == cls_to_aug:
                                obj_added_for_target_class = True

                        if obj_added_for_target_class:
                            augmented_count_for_this_class += 1
                            pbar.update(1)
                            pbar.set_description(
                                f"🧬 Class {cls_to_aug} 증강 중 ({final_counts_after_aug.get(cls_to_aug, 0)}/{target_per_class})"
                            )
                except Exception as e_write:
                    print(f"⚠️ Class {cls_to_aug}: 증강된 파일 저장 중 오류 ({new_file_base_name}): {e_write}")
                    if os.path.exists(output_augmented_image_path):
                        os.remove(output_augmented_image_path)
                    if os.path.exists(output_augmented_label_path):
                        os.remove(output_augmented_label_path)
                    continue

            pbar.close()
            if augmented_count_for_this_class < num_needed_for_class:
                print(
                    f"🔔 Class {cls_to_aug}: 목표 수량({target_per_class}개) 중 "
                    f"{final_counts_after_aug.get(cls_to_aug, 0)}개까지 증강 완료 "
                    "(시도 횟수 초과 또는 후보 부족)."
                )

        print("\n✅ 모든 지정된 클래스에 대한 증강 작업 완료.")
    elif args.aug:
        print("\n🛑 지정된 혹은 감지된 증강 대상 클래스가 없어 증강 작업을 수행하지 않았습니다.")

# ───────────────────────────────
# 스크립트 실행 부분
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="YOLO 데이터셋의 특정 클래스를 오버샘플링하고 이미지를 증강합니다."
    )
    parser.add_argument('--aug', nargs='+', type=str,
                        help='증강할 클래스 번호 리스트 또는 "all"')
    parser.add_argument('--target', type=int,
                        help='클래스당 목표 수량 (증강 시 필수)')
    parser.add_argument('--train_copy', action='store_true',
                        help='증강 작업 전, 원본 학습 데이터를 증강용 폴더로 복사합니다.')
    parser.add_argument('--list', nargs='?', const=True, default=None,
                        help='현재 증강 폴더의 클래스 목록 및 수량만 출력합니다. (예: --list 또는 --list 50)')

    args = parser.parse_args()

    # 필수 인자 체크 (증강 모드일 경우 --target 확인)
    if args.aug and args.target is None:
        parser.error("❌ 오류: --aug 옵션 사용 시 --target 값을 반드시 지정해야 합니다. 예: --target 100")

    # 메인 로직 함수 호출
    main_logic(args)
