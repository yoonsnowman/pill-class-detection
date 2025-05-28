import os
import shutil
import random

def split_and_copy_images_to_train_val(
    source_image_dir,
    train_output_dir,
    val_output_dir,
    split_ratio=0.8
):
    """
    주어진 원본 이미지 디렉토리의 이미지를 훈련(train)과 검증(val) 세트로 분리하여
    지정된 출력 디렉토리에 복사합니다.

    Args:
        source_image_dir (str): 원본 이미지 파일들이 있는 디렉토리 (예: 'data/train_images').
        train_output_dir (str): 훈련 이미지들을 저장할 디렉토리 (예: 'data/train_images_split').
        val_output_dir (str): 검증 이미지들을 저장할 디렉토리 (예: 'data/val_images').
        split_ratio (float): 훈련 세트의 비율 (0.0 ~ 1.0 사이). 기본값은 0.8 (80%).
    """
    # 출력 디렉토리 생성 또는 기존 디렉토리 비우기
    for directory in [train_output_dir, val_output_dir]:
        if os.path.exists(directory):
            print(f"⚠️ 기존 디렉토리 '{directory}'를 비웁니다...")
            shutil.rmtree(directory) # 기존 내용 삭제
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 디렉토리 생성/확인: {directory}")

    # 이미지 파일 목록 가져오기
    all_images = []
    print(f"\n{source_image_dir}에서 이미지 파일 목록을 가져오는 중...")
    for filename in os.listdir(source_image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            all_images.append(filename)
    
    if not all_images:
        print(f"❌ 오류: '{source_image_dir}' 디렉토리에서 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    random.shuffle(all_images) # 이미지 목록을 무작위로 섞음

    # 분리 지점 계산
    num_train = int(len(all_images) * split_ratio)
    train_images = all_images[:num_train]
    val_images = all_images[num_train:]

    print(f"총 {len(all_images)}개의 이미지 중:")
    print(f"  - 훈련 세트: {len(train_images)}개 ({split_ratio*100:.0f}%)")
    print(f"  - 검증 세트: {len(val_images)}개 ({(1-split_ratio)*100:.0f}%)")

    # 훈련 이미지 복사
    print(f"\n훈련 이미지들을 '{train_output_dir}'로 복사 중...")
    for img_filename in tqdm(train_images, desc="훈련 이미지 복사"):
        src_path = os.path.join(source_image_dir, img_filename)
        dst_path = os.path.join(train_output_dir, img_filename)
        shutil.copy2(src_path, dst_path) # 메타데이터도 함께 복사

    # 검증 이미지 복사
    print(f"\n검증 이미지들을 '{val_output_dir}'로 복사 중...")
    for img_filename in tqdm(val_images, desc="검증 이미지 복사"):
        src_path = os.path.join(source_image_dir, img_filename)
        dst_path = os.path.join(val_output_dir, img_filename)
        shutil.copy2(src_path, dst_path) # 메타데이터도 함께 복사

    print("\n✅ 이미지 분리 및 복사 완료!")

# --- 사용 예시 (이 부분을 실행합니다) ---
if __name__ == "__main__":
    # 데이터셋 경로 설정
    source_images_folder = 'data/train_images'      # 원본 이미지 폴더
    train_output_folder = 'data/train_images_split' # 훈련 이미지 저장 폴더
    val_output_folder = 'data/val_images'           # 검증 이미지 저장 폴더

    # 스크립트 실행
    split_and_copy_images_to_train_val(
        source_image_dir=source_images_folder,
        train_output_dir=train_output_folder,
        val_output_dir=val_output_folder,
        split_ratio=0.8 # 80%를 훈련 세트로 사용
    )