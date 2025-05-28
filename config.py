####################################################
# 내용: 코랩 연동 directory 목록
# 작성자: 윤승호
# 수정일: 2025. 05. 28. 22:00
# 용도: 로컬 경로와 구글 드라이브 경로 연동
####################################################


# {환경명 : 로컬 경로} 형식으로 기입
dirs = {
    'yolo_train':    'data/yolo/train_images',
    'yolo_val':      'data/yolo/val_images',
    'yolo_test':     'data/yolo/test_images',
    'yolo_anns':     'data/yolo/train_annotations',
    'yolo_labels':   'data/yolo/processed/labels',
    'yolo_classes':  'data/yolo/processed/classes.txt',

    'detr_train':    'data/detr/train_images',
    'detr_val':      'data/detr/val_images',
    'detr_test':     'data/detr/test_images',
    'detr_anns':     'data/detr/train_annotations',
    'detr_labels':   'data/detr/processed/labels',
    'detr_classes':  'data/detr/processed/classes.txt',

    'font':          'data/font/NanumGothic.ttf',
    'trainset':      'data/train',
    'valset':        'data/val',
    'testset':       'data/test',
}

import os, sys
def develop(environment='base'):
    local_path = os.path.abspath(os.path.dirname(__file__))
    cloud_path = '/content/drive/MyDrive/dev/projects/project1/pill-detect-ai'
    cloud_env_path = f'/content/drive/MyDrive/dev/environment/{environment}'

    if 'google.colab' in sys.modules:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print('⚠️ Colab에서 drive.mount() 실패:', e)

        # environment 폴더 없으면 생성
        os.makedirs(cloud_env_path, exist_ok=True)

        # PYTHONPATH 우선순위로 추가
        if cloud_env_path not in sys.path:
            sys.path.insert(0, cloud_env_path)

        return {k: os.path.normpath(os.path.join(cloud_path, v)) for k, v in dirs.items()}

    else:
        # 로컬 환경일 때: 현재 디렉토리 기준 상대경로
        print(f"루트 디렉토리: {local_path}")
        return {k: os.path.normpath(os.path.join(local_path, v)) for k, v in dirs.items()}
        
DIR = develop()



# 터미널에서 직접 실행하면 경로 확인 출력
if __name__ == "__main__":
        import torch
        if torch.cuda.is_available():
            print(f"⭕ CUDA 사용 가능 ▶ 현재 장치: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ CUDA 사용 불가 ▶ 현재 장치: CPU")

        print("\n📁 [클라우드 연동 경로]")
        for key, path in DIR.items():
            print(f"{key:15}: {path}")