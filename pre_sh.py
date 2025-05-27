""" 개발환경 설정 """
def develop(environment='base'):
    n1, d1 = 'base',    'data/project1data'
    n2, d2 = 'train',   'data/project1data/raw/train_images'
    n3, d3 = 'test',    'data/project1data/raw/test_images'
    n4, d4 = 'font',    'data/project1data/raw/NanumGothic.ttf'
    n5, d5 = 'labels',  'data/project1data/raw/processed/labels'
    n6, d6 = 'classes', 'data/project1data/raw/processed/classes.txt'
    dirs = {n1:d1, n2:d2, n3:d3, n4:d4, n5:d5, n6:d6}

    local_path = 'G:/내 드라이브/dev/'
    cloud_path = '/content/drive/MyDrive/dev'
    env_path = f'{cloud_path}/environment/{environment}'

    import os, sys
    if 'google' in sys.modules:
        from google.colab import drive; drive.mount('/content/drive')
        if env_path not in sys.path: sys.path.insert(0, env_path)
        os.environ['TORCH_HOME'] = f'{cloud_path}/model/pytorch_model'
        return {k: os.path.join(cloud_path, v) for k, v in dirs.items()}
    else: return {k: os.path.join(local_path, v) for k, v in dirs.items()}
for k, v in develop().items(): print(f"{k:10}: {v}")
DIR = develop()


""" 라이브러리 로드 """
# 표준 라이브러리
import os, shutil
import random
import json
from collections import defaultdict, Counter
# 수치/시각화
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# PIL
from PIL import Image, ImageDraw, ImageFont
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.cuda.amp import GradScaler, autocast
from torchinfo import summary
# 데이터 로더
from torch.utils.data import Dataset, DataLoader
# torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2
# 진행 표시
from tqdm import tqdm
# 머신러닝 평가 지표
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
# 데이터 증강
import albumentations as A
from albumentations.pytorch import ToTensorV2


""" 데이터 전처리 """
# Helper 클래스
class PillAnnotationParser:
    def __init__(self, annotation_root, output_dir):
        self.annotation_root = annotation_root
        self.output_dir = output_dir
        self.image_data = []
        self.annotations = []
        self.categories = {}
        self.class_map = {}

    def _find_json_files(self):
        json_paths = []
        for root, _, files in os.walk(self.annotation_root):
            for file in files:
                if file.endswith('.json'):
                    json_paths.append(os.path.join(root, file))
        print(f"🔍 JSON 파일 {len(json_paths)}개 발견됨")
        return json_paths

    def load_annotations(self):
        print("📂 어노테이션 로드 중...")
        json_files = self._find_json_files()

        for path in tqdm(json_files, desc="📑 JSON 처리 중"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'images' in data and 'annotations' in data:
                        self.image_data.extend(data['images'])
                        self.annotations.extend(data['annotations'])
                        for cat in data.get('categories', []):
                            self.categories[cat['id']] = cat['name']
            except Exception as e:
                print(f"❌ 오류 발생: {path}")
                print(f"    {e}")
        print(f"✅ 이미지 {len(self.image_data)}개, 어노테이션 {len(self.annotations)}개 로드 완료")

    def generate_class_map(self):
        print("🧠 전체 클래스 맵 생성 중...")
        class_ids = set(ann['category_id'] for ann in self.annotations)
        self.class_map = {cat_id: idx for idx, cat_id in enumerate(sorted(class_ids))}
        print(f"✅ 총 {len(self.class_map)}개 클래스 매핑 완료!")


    def save_yolo_annotations(self):
        labels_dir = os.path.join(self.output_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        print("🚀 YOLO 라벨 저장 시작...")

        image_id_map = {img['id']: img for img in self.image_data}
        label_files = defaultdict(list)
        count = 0

        for ann in self.annotations:
            cat_id = ann['category_id']
            if cat_id not in self.class_map:
                continue

            img_info = image_id_map[ann['image_id']]
            w, h = img_info['width'], img_info['height']
            x, y, bw, bh = ann['bbox']

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            w_norm = bw / w
            h_norm = bh / h
            class_idx = self.class_map[cat_id]

            label_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            label_files[img_info['file_name']].append(label_line)
            count += 1

        for filename, lines in label_files.items():
            txt_name = os.path.splitext(filename)[0] + '.txt'
            with open(os.path.join(labels_dir, txt_name), 'w') as f:
                f.write('\n'.join(lines))

        print(f"✅ 총 {count}개 어노테이션 저장 완료! ({labels_dir})")

    def export_class_map(self):
        class_txt_path = os.path.join(self.output_dir, 'classes.txt')
        with open(class_txt_path, 'w') as f:
            sorted_items = sorted(self.class_map.items(), key=lambda x: x[1])
            for cat_id, new_id in sorted_items:
                name = self.categories.get(cat_id, 'unknown')
                f.write(f"{new_id}: {name}\n")
        print(f"📝 클래스 맵 저장 완료: {class_txt_path}")

# # 🏁 실행 코드
# if __name__ == "__main__":
#     parser = PillAnnotationParser(
#         annotation_root='/content/drive/MyDrive/dev/data/project1data/raw/train_annotations',
#         output_dir='/content/drive/MyDrive/dev/data/project1data/raw/processed'
#     )

#     parser.load_annotations()
#     parser.generate_class_map()
#     parser.save_yolo_annotations()
#     parser.export_class_map()


""" BBOX 매핑 확인 시각화 """
# 시각화
def load_class_names(path):
    class_names = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            _, name = line.strip().split(': ')
            class_names.append(name)
    return class_names

def draw_yolo_bbox_on_image(image_path, label_path, class_names, font_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    try:
        font = ImageFont.truetype(font_path, 30)
    except:
        font = ImageFont.load_default()

    if not os.path.exists(label_path):
        return image  # 그냥 이미지만 반환

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])

        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

        label = class_names[cls] if class_names else f"{cls}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x1
        text_y = y1 - text_height - 6
        if text_y < 0:
            text_y = y1 + 4

        draw.rectangle(
            [text_x, text_y, text_x + text_width + 6, text_y + text_height + 4],
            fill="black"
        )
        draw.text((text_x + 3, text_y + 2), label, font=font, fill="lime")

    return image

def show_yolo_images_grid(image_dir, label_dir, class_names, font_path, count=9):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    random.shuffle(image_files)

    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.png', '.txt'))

        if os.path.exists(label_path):
            img = draw_yolo_bbox_on_image(img_path, label_path, class_names, font_path)
            images.append(img)
        if len(images) == count:
            break

    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_yolo_images_grid(DIR['train'], DIR['labels'], load_class_names(DIR['classes']), DIR['font'], count=9)




""" 데이터셋 스플릿 """
# Dataset 클래스
class PillDataset(Dataset):
    def __init__(self, image_dir, label_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1) 이미지, bbox·label 로드
        name    = self.files[idx]
        img_np  = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        lbl_path= os.path.join(self.label_dir, name.replace('.png','.txt'))
        boxes, labels = [], []
        if os.path.exists(lbl_path):
            for line in open(lbl_path):
                cls, x_c, y_c, w, h = map(float, line.split())
                labels.append(int(cls)); boxes.append([x_c, y_c, w, h])

        # 2) Albumentations transform 호출 (image, bboxes, class_labels 모두 넘겨줌)
        if self.transform:
            data = self.transform(image=img_np, bboxes=boxes, class_labels=labels)
            image  = data['image']
            boxes  = data['bboxes']
            labels = data['class_labels']
        else:
            image = ToTensorV2()(image=img_np)['image']  # fallback

        # 3) tensor로 변환
        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels,dtype=torch.int64)
        }
        return image, target

# 데이터 증강
train_transform = A.Compose([
    A.Resize(640, 640),
    A.Affine(translate_percent=(0.1, 0.1), scale=(0.9, 1.1), rotate=15, shear=5, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

test_transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 데이터 스플릿
all_imgs = [f for f in os.listdir(DIR['train']) if f.endswith('.png')]
train_files, val_files = train_test_split(all_imgs, test_size=0.2, random_state=42)
test_files = [f for f in os.listdir(DIR['test']) if f.endswith('.png')]

train_dataset = PillDataset(image_dir=DIR['train'], label_dir=DIR['labels'], file_list=train_files, transform=train_transform)
val_dataset   = PillDataset(image_dir=DIR['train'], label_dir=DIR['labels'], file_list=val_files,   transform=test_transform)
test_dataset  = PillDataset(image_dir=DIR['test'],  label_dir=DIR['labels'], file_list=test_files, transform=test_transform)

train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True,  collate_fn=lambda batch: tuple(zip(*batch)))
val_loader    = DataLoader(val_dataset,   batch_size=16, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))
test_loader   = DataLoader(test_dataset,  batch_size=16, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

imgs, targets = next(iter(train_loader))
imgs = torch.stack(imgs)         # list/tuple → (B, C, H, W) tensor
print(imgs.shape)                # e.g. torch.Size([16, 3, 640, 640])
print([t['boxes'].shape for t in targets])



""" 모델 인풋 데이터 시각화 """
# 시각화
def show_augmented_bbox_images(dataset, count=9):
    indices = random.sample(range(len(dataset)), count)
    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    for ax, idx in zip(axes.flat, indices):
        img_tensor, target = dataset[idx]
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = img * 0.5 + 0.5
        h, w, _ = img.shape
        ax.imshow(img)
        for x_c, y_c, bw, bh in target['boxes'].cpu().numpy():
            x1 = (x_c - bw/2) * w
            y1 = (y_c - bh/2) * h
            rect = patches.Rectangle((x1, y1), bw * w, bh * h,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_augmented_bbox_images(train_dataset)
