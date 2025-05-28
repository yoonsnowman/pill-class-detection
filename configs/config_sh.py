# config/config_sh.py
import os
import sys
from configs.config_dirs import get_dirs

"""
★ config_이니셜.py를 수정하지 말고 import해서 경로만 사용
★ 로컬/코랩 환경 자동 인식
★ 폴더 구조는 프로젝트 루트를 기준으로 동일하게 유지해야 실행됩니다.
★ 로컬에서는 폴더 열기 - 깃허브 저장소 하면 끝
★ 코랩에서는 프로젝트 저장 경로 통일 필요
★ train_dir = DIR['raw_train'] 이런 식으로 호출해서 사용
"""


def develop(environment='base'):
    # 로컬 기준 루트 경로 (현재 config.py 기준)
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"루트 디렉토리: {local_path}")
    # 코랩 기준 루트 경로 (Google Drive 기준)
    cloud_path = '/content/drive/MyDrive/dev/projects/project1/pill-detect-ai'
    cloud_env_path = '/content/drive/MyDrive/dev/environment'

    # 상대경로 정의 (공통)
    dirs = get_dirs(environment)

    if 'google.colab' in sys.modules:
        # 코랩 환경일 때: 드라이브 마운트 및 경로 설정
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print('⚠️ Colab에서 drive.mount() 실패:', e)

        # ⬇️ environment 폴더 없으면 생성
        os.makedirs(cloud_env_path, exist_ok=True)

        # ⬇️ PYTHONPATH 우선순위로 추가
        if cloud_env_path not in sys.path:
            sys.path.insert(0, cloud_env_path)

        return {k: os.path.join(cloud_path, v) for k, v in dirs.items()}


    else:
        # 로컬 환경일 때: 현재 디렉토리 기준 상대경로
        return {k: os.path.join(local_path, v) for k, v in dirs.items()}

# 전역에서 쓸 수 있는 DIR 객체
DIR = develop()

# 🔽 터미널에서 직접 실행하면 경로 확인 출력
if __name__ == "__main__":
    print("📁 [자동 생성된 경로 확인]")
    for key, path in DIR.items():
        print(f"{key:15}: {path}")
