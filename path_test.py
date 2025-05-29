import os, sys, glob

# 이미지 개수 세는 함수
def count_images(folder_path):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    count = 0
    for ext in exts:
        count += len(glob.glob(os.path.join(folder_path, ext)))
    return count

print("\n📂 [이미지 수량 확인]")

for name in ['train_images', 'test_images']:
    dir_path = f'data/yolo/{name}'
    if not os.path.exists(dir_path):
        print(f"- {name} 경로 존재하지 않음 ❌ → {dir_path}")
    else:
        num_images = count_images(dir_path)
        print(f"- {name}: {num_images}장 ({dir_path})")
