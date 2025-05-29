import os, shutil
from sklearn.model_selection import train_test_split


def create_yolo_dataset_split(train_files, val_files, test_files):
    base = 'data/yolo/raw_split'
    src_img_train = 'data/yolo/train_images'     # 원본 학습 이미지
    src_img_test  = 'data/yolo/test_images'      # 원본 테스트 이미지
    src_labels    = 'data/yolo/raw_split/labels' # YOLO 라벨 폴더

    target_struct = {
        'train': {'files': train_files, 'img_src': src_img_train},
        'val':   {'files': val_files,   'img_src': src_img_train},
        'test':  {'files': test_files,  'img_src': src_img_test},
    }

    for split, cfg in target_struct.items():
        img_dst = os.path.join(base, split, 'images')
        lbl_dst = os.path.join(base, split, 'labels')
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        img_copied = img_skipped = lbl_copied = lbl_skipped = 0

        for fname in cfg['files']:
            # 이미지 복사
            src_img_path = os.path.join(cfg['img_src'], fname)
            dst_img_path = os.path.join(img_dst, fname)
            if os.path.exists(dst_img_path):
                img_skipped += 1
            elif os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
                img_copied += 1

            # 라벨 복사
            lbl_name = fname.replace('.png', '.txt')
            src_lbl_path = os.path.join(src_labels, lbl_name)
            dst_lbl_path = os.path.join(lbl_dst, lbl_name)
            if os.path.exists(dst_lbl_path):
                lbl_skipped += 1
            elif os.path.exists(src_lbl_path):
                shutil.copy(src_lbl_path, dst_lbl_path)
                lbl_copied += 1

        print(f'📁 [{split.upper()}] images: 복사 완료 {img_copied}개 / 이미 존재 {img_skipped}개')
        print(f'📁 [{split.upper()}] labels: 복사 완료 {lbl_copied}개 / 이미 존재 {lbl_skipped}개\n')


if __name__ == '__main__':
    train_img_dir = 'data/yolo/train_images'
    test_img_dir  = 'data/yolo/test_images'

    # 🔍 목록 불러오기 (존재하는 이미지 기준)
    all_imgs = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    train_files, val_files = train_test_split(all_imgs, test_size=0.2, random_state=42)

    test_files = [f for f in os.listdir(test_img_dir) if f.endswith('.png')]

    # ✅ 스플릿 및 복사 실행
    create_yolo_dataset_split(train_files, val_files, test_files)
