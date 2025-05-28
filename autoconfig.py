####################################################
# 내용: 로컬 경로와 구글 드라이브 경로 자동 연동
# 작성자: 윤승호
# 수정일: 2025. 05. 28. 22:00
# 용도: 로컬 경로와 구글 드라이브 경로 연동
####################################################



# autoconfig.py
# {환경명 : 로컬 경로} 형식으로 기입
import os, sys

def develop(environment='base'):
    local_root = os.path.abspath(os.path.dirname(__file__))
    cloud_root = '/content/drive/MyDrive/dev/projects/project1/pill-detect-ai'
    cloud_env_path = f'/content/drive/MyDrive/dev/environment/{environment}'

    if 'google.colab' in sys.modules:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print('⚠️ Colab에서 drive.mount() 실패:', e)

        os.makedirs(cloud_env_path, exist_ok=True)
        if cloud_env_path not in sys.path:
            sys.path.insert(0, cloud_env_path)

        return lambda rel_path: os.path.normpath(os.path.join(cloud_root, rel_path))

    else:
        print(f"루트 디렉토리: {local_root}")
        return lambda rel_path: os.path.normpath(os.path.join(local_root, rel_path))

DIR = develop()

# 터미널에서 실행 시 GPU 사용 여부 확인
if __name__ == "__main__":
        import torch
        if torch.cuda.is_available():
            print(f"CUDA 가능여부: ⭕ (현재 장치: {torch.cuda.get_device_name(0)})")
        else:
            print("CUDA 가능여부: ❌ (현재 장치: CPU)")
