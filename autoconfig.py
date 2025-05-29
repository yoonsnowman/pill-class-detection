#################################################################
# autoconfig.py / v1.3.11
# 사용 방법
# import os, sys  # .ipynb 파일 폴더 깊이 1개 시 추가
# sys.path.append(os.path.abspath('..'))  # .ipynb 파일 폴더 깊이 1개 시 추가
# from autoconfig import DIR
#################################################################

import os, sys

# 이 파일이 있는 디렉토리를 sys.path에 등록
try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Jupyter 환경에서 코드 직접 실행 시 __file__이 없으므로 현재 작업 디렉토리로 대체
    ROOT_DIR = os.getcwd()

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def develop(environment='base'):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()  # Jupyter 환경 대응

    local_root = base_dir
    cloud_root = '/content/drive/MyDrive/dev/projects/project1/pill-detect-ai'
    cloud_env_path = f'/content/drive/MyDrive/dev/environment/{environment}'

    if 'google.colab' in sys.modules:
        os.makedirs(cloud_env_path, exist_ok=True)
        if cloud_env_path not in sys.path:
            sys.path.insert(0, cloud_env_path)

        # Colab 절대경로 반환
        return lambda rel_path: os.path.normpath(os.path.join(cloud_root, rel_path)).replace("\\", "/")
    else:
        print(f"루트 디렉토리: {local_root}")

        # 로컬 상대경로 반환
        return lambda rel_path: os.path.relpath(os.path.normpath(os.path.join(local_root, rel_path)), start=local_root).replace("\\", "/")

DIR = develop()


# 터미널 실행 시 CUDA 사용 여부 출력
if __name__ == "__main__":        
        import torch
        if torch.cuda.is_available():
            print(f"CUDA 가능여부: ⭕ (현재 장치: {torch.cuda.get_device_name(0)})")
        else:
            print("CUDA 가능여부: ❌ (현재 장치: CPU)")

