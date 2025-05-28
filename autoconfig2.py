# autoconfig.py
import os, sys

def develop(environment='base', safe=False):
    # 🟢 경로 설정
    colab_root = '/content/drive/MyDrive/dev/projects/project1/pill-detect-ai'
    colab_env_path = f'/content/drive/MyDrive/dev/environment/{environment}'

    local_root_real = os.path.abspath(os.path.dirname(__file__))  # G:\내 드라이브\dev\...
    local_root_fake = local_root_real.replace('내 드라이브', '내드라이브')             # G:\내드라이브\dev\...

    local_env_path_real = f'{local_root_real}\\environment\\{environment}'
    local_env_path_fake = f'{local_root_fake}\\environment\\{environment}'

    # 🟠 Colab
    if 'google.colab' in sys.modules:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print('⚠️ Colab에서 drive.mount() 실패:', e)

        os.makedirs(colab_env_path, exist_ok=True)
        if colab_env_path not in sys.path:
            sys.path.insert(0, colab_env_path)

        return lambda rel_path: os.path.normpath(os.path.join(colab_root, rel_path))

    # 🔵 Local
    else:
        print(f"루트 디렉토리: {local_root_real}")

        # 심볼릭 링크 생성 (공백 제거용)
        target = r"G:\내 드라이브"
        alias  = r"G:\내드라이브"
        if not os.path.exists(alias):
            try:
                os.system(f'mklink /D "{alias}" "{target}"')
                print(f"🔗 심볼릭 링크 생성됨: {alias} → {target}")
            except Exception as e:
                print(f"❌ 심볼릭 링크 생성 실패: {e}")

        # sys.path 등록 (안전 경로 기준)
        if safe:
            if local_env_path_fake not in sys.path:
                sys.path.insert(0, local_env_path_fake)
            return lambda rel_path: os.path.normpath(os.path.join(local_root_fake, rel_path))
        else:
            if local_env_path_real not in sys.path:
                sys.path.insert(0, local_env_path_real)
            return lambda rel_path: os.path.normpath(os.path.join(local_root_real, rel_path))


# ⚙️ 일반 경로 (공백 있음, 일반 작업용)
DIR = develop()

# ⚙️ 안전 경로 (공백 없음, YOLO 등 외부 라이브러리용)
DIR_SAFE = develop(safe=True)

# ✅ 단독 실행 시: GPU 확인
if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        print(f"CUDA 가능여부: ⭕ (현재 장치: {torch.cuda.get_device_name(0)})")
    else:
        print("CUDA 가능여부: ❌ (현재 장치: CPU)")
