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
# 데이터 로더
from torch.utils.data import Dataset, DataLoader
# torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2
# 진행 표시
from tqdm import tqdm




""" 데이터 전처리 """
# Helper 클래스
class PillAnnotationParser:
    def __init__(self, annotation_dir, output_dir):
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.image_data = []
        self.annotations = []
        self.categories = {}
        self.class_map = {}

    def _find_json_files(self):
        json_paths = []
        for root, _, files in os.walk(self.annotation_dir):
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


# 🏁 실행 코드
if __name__ == "__main__":
    parser = PillAnnotationParser(
        annotation_dir = 'data/yolo/train_annotations',
        output_dir = 'data/yolo/raw_split'
    )

    parser.load_annotations()
    parser.generate_class_map()
    parser.save_yolo_annotations()
    parser.export_class_map()

