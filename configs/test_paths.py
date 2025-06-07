import os, glob
import configs.config_paths as cc

# 📂 루트 기준 경로
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 📌 이미지 수량 세기
def count_images(folder_path):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    return sum(len(glob.glob(os.path.join(folder_path, ext))) for ext in exts)

# 📌 폴더 내 항목 수 세기
def count_entries(path):
    if not os.path.exists(path):
        return 0, 0
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    dirs  = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return len(files), len(dirs)

# -------------------------
print("\n📂 [이미지 수량 확인]")

img_dirs = {
    'train': os.path.join(cc.PRE_IN_DIR,'train_images'),
    'test': os.path.join(cc.PRE_IN_DIR,'test_images'),
}

for name, path in img_dirs.items():
    if not os.path.exists(path):
        print(f"- {name} 이미지 경로 ❌ 없음 → {path}")
    else:
        print(f"- {name}: {count_images(path)}장 ({path})")

# -------------------------
print("\n🧭 [프로젝트 구조 확인]")

for folder_name in ['scripts', 'src']:
    path = os.path.join(ROOT_DIR, folder_name)
    files, dirs = count_entries(path)
    print(f"- {folder_name}/ → 📄 {files}개, 📁 {dirs}개 (경로: {path})")
