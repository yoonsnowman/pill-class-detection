import os
import shutil
from collections import Counter, defaultdict
from tqdm import tqdm
import cv2
import albumentations as A
import random
import argparse
import statistics


"""
🔧 사용법:

# ─── 기존 train 이미지 복사 ───────────────────────────────
!python scripts/oversample_rare_class.py --train_copy

# ─── 전체 클래스 개수 출력 + 50개 미만 클래스 추가 확인 ────────
!python scripts/oversample_rare_class.py --list 50

# ─── 전체 클래스 50장이 되도록 증강 ─────────────────────────
!python scripts/oversample_rare_class.py --aug all --target 50

# ─── 특정 클래스만 증강 50장까지 증강 ────────────────────────
!python scripts/oversample_rare_class.py --aug 2 5 8 --target 50

"""


# ───────────────────────────────
# argparse 설정
parser = argparse.ArgumentParser()
parser.add_argument('--aug', nargs='+', type=str, help='증강할 클래스 번호 리스트 또는 "all"')
parser.add_argument('--target', type=int, help='클래스당 목표 수량 (필수)')
parser.add_argument('--train_copy', action='store_true', help='기존 train 이미지+라벨 복사 여부')
parser.add_argument('--list', nargs='?', const=True, help='클래스 목록 출력만 (opt: --list 또는 --list 50)')
args = parser.parse_args()

# ───────────────────────────────
# 필수 인자 체크
if not args.list and not args.train_copy and args.target is None:
    print("❌ 오류: --target 값을 반드시 지정해야 합니다. 예: --target 100")
    exit()

# ───────────────────────────────
# 경로 설정
base_label_dir = 'data/yolo/pill_yolo_format/labels/train'
aug_label_dir  = 'data/yolo/pill_yolo_format/labels/train_aug'
base_image_dir = 'data/yolo/pill_yolo_format/images/train'
aug_image_dir  = 'data/yolo/pill_yolo_format/images/train_aug'

os.makedirs(aug_label_dir, exist_ok=True)
os.makedirs(aug_image_dir, exist_ok=True)

# ───────────────────────────────
# 증강기 정의
augmentor = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=3, val_shift_limit=0, p=0.3),
    A.Affine(scale=(0.95, 1.05), p=0.5)
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))

# ───────────────────────────────
# 이미지 확장자 확인 함수
def find_image_file(base_name):
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(base_image_dir, base_name + ext)
        if os.path.exists(path):
            return path
    return None

# ───────────────────────────────
# 클래스 수량 세기
class_counts = Counter()
file_map = defaultdict(list)

for file in tqdm(os.listdir(aug_label_dir), desc='📊 라벨 수집 중'):
    if not file.endswith('.txt'):
        continue
    with open(os.path.join(aug_label_dir, file)) as f:
        lines = f.readlines()
        classes = [int(float(line.strip().split()[0])) for line in lines]
        for cls in classes:
            class_counts[cls] += 1
        for cls in set(classes):
            file_map[cls].append(file)

# ───────────────────────────────
# --list 모드
if args.list is not None:
    print("\n📊 전체 클래스별 이미지 수:")
    for cls in sorted(class_counts):
        print(f"Class {cls:2d}: {class_counts[cls]}장")

    values = list(class_counts.values())
    mode_val = statistics.mode(values)
    median_val = int(statistics.median(values))
    rare_classes = sorted([cls for cls, count in class_counts.items() if count < median_val])

    print(f"\n📌 mode (최빈값): {mode_val}")
    print(f"📌 median (중앙값): {median_val}")
    print(f"📉 median 미만 클래스: {' '.join(map(str, rare_classes))}")

    if isinstance(args.list, str) and args.list.isdigit():
        threshold = int(args.list)
        below_threshold = sorted([cls for cls, count in class_counts.items() if count < threshold])
        print(f"\n📉 {threshold}개 미만 클래스: {' '.join(map(str, below_threshold))}")
    exit()

# ───────────────────────────────
# --train_copy 옵션 시 복사
if args.train_copy:
    for file in tqdm(os.listdir(base_label_dir), desc='📥 원본 train 복사 중'):
        base = os.path.splitext(file)[0]
        image_path = find_image_file(base)
        if image_path:
            shutil.copy(image_path, os.path.join(aug_image_dir, os.path.basename(image_path)))
            shutil.copy(os.path.join(base_label_dir, file), os.path.join(aug_label_dir, file))

# ───────────────────────────────
# 증강 클래스 결정
if args.aug and args.aug[0].lower() == 'all':
    target_classes = sorted(class_counts.keys())
elif args.aug:
    try:
        target_classes = list(map(int, args.aug))
    except ValueError:
        print("❌ 클래스 번호는 정수로 입력하거나 'all'을 사용해야 합니다.")
        exit()
else:
    target_classes = []

target_per_class = args.target
final_counts = class_counts.copy()

# ───────────────────────────────
# 증강 시작
if target_classes:
    print(f"\n🧪 증강 시작 (목표: 클래스당 최대 {target_per_class}장)\n")

    for cls in target_classes:
        current = final_counts[cls]
        if current >= target_per_class:
            continue

        needed = target_per_class - current
        candidates = file_map[cls]
        pbar = tqdm(total=needed, desc=f"🔁 Class {cls} 증강 중")

        i, attempts = 0, 0
        max_attempts = needed * 5

        while i < needed and attempts < max_attempts:
            src_label_file = random.choice(candidates)
            base_name = os.path.splitext(src_label_file)[0]
            image_path = find_image_file(base_name)
            if not image_path:
                attempts += 1
                continue

            image = cv2.imread(image_path)
            with open(os.path.join(aug_label_dir, src_label_file)) as f:
                lines = f.readlines()
                bboxes = []
                class_labels = []
                for line in lines:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    bboxes.append(bbox)
                    class_labels.append(cls_id)

            try:
                augmented = augmentor(image=image, bboxes=bboxes, class_labels=class_labels)
            except:
                attempts += 1
                continue

            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            new_base = f"{base_name}_aug_{i}"
            ext = os.path.splitext(image_path)[1]
            out_img_path = os.path.join(aug_image_dir, new_base + ext)
            out_lbl_path = os.path.join(aug_label_dir, new_base + '.txt')

            cv2.imwrite(out_img_path, aug_image)
            with open(out_lbl_path, 'w') as f:
                for cid, bbox in zip(aug_labels, aug_bboxes):
                    f.write(f"{int(cid)} {' '.join(f'{v:.6f}' for v in bbox)}\n")

            final_counts[cls] += 1
            i += 1
            attempts += 1
            pbar.update(1)

        pbar.close()
