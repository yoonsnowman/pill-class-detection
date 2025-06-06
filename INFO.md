**Project Version 1.4.0**

---
>data/yolo/ 경로에 프로젝트 압축 파일 해제 후 사용하시기 바랍니다.
>train_images, train_annotations, test_images 폴더가 존재해야 합니다.

## 1. 명령어 사용 방법
### main.py
기본 셋팅값으로 전처리, 학습, 추론 실행
- `python -m main`
설정값 사용(모델pt명, outputs/에 저장할 폴더명, 배치, 에포크)
- `python -m main all --model yolov8n --name v8n --b 32 --e 10`


### configs/config_paths.py
- `import configs.config_paths as cc` 스크립트에 추가
- 스크립트에서 `cc.PRE_IN_DIR` 등을 사용하여 경로 호출


### src/train.py
수동 셋팅값 사용(모델pt명, outputs/에 저장할 폴더명, 배치, 에포크)
- `python -m src.train --model yolov8n --name v8n --b 16 --e 10`


### src/preprocess.py
명령어 특이사항 없음, 첫 실행 시 40분 가량 소요
- `python -m src.preprocess`


### src/predict.py
output/에 저장할 폴더명 명령어에 지정
- `python -m src.predict --name myfoldername`


### scripts/class_counter.py
기본 탐색 폴더: data/yolo/labels/train/
데이터 증강 한 폴더 탐색 시 스크립트 내에서 폴더명과 파일명 변경 필요
- `python -m scripts.class_counter`


### scripts/oversample.py
train 이미지 증강 폴더로 복사(필수)
- `python -m scripts.oversample --train_copy`
특정 수량 이하 클래스 출력
- `python -m scripts.oversample --list 50`
전체 클래스 -> 목표 수량까지 증강
- `python -m scripts.oversample --aug all --target 50`
특정 클래스 -> 목표 수량까지 증강
- `python -m scripts.oversample --aug 2 5 8 --target 50`



---
## 2. 업데이트 내용
### 변경사항
- 모든 파일 `config_paths.py` 경로 테스트 완료
- `main.py` 추가로 `src/` 내 전체 코드(전처리, 학습, 추론) 한번에 실행 가능


### 진행예정
- `notebooks/`에 EDA 노트북 추가 예정
- `main.py`에서 `src/` 내 전체 코드 실행 예정
- `models/` 폴더 내 모델별 정보 `.txt`로 정리 예정
- `preprocess.py` 주석 처리 및 경로 설정 업데이트 예정


### 적용완료
- 모든 경로를 `config_paths.py`에 통합 → `import configs.config_paths as cc` 형태로 호출
- `data/detr/` 디렉토리 제거
- 모든 학습 결과 → `outputs/` 폴더로 통합 관리
- `scripts/` 폴더 생성:
    - `class_counter.py`: 클래스 수 집계
    - `oversample.py`: 소수 클래스 증강
- `src/` 폴더 생성:
    - `preprocess.py`, `train.py`, `predict.py` 작성 (공통 로직 분리)
- `.gitignore` 업데이트:
    - `weights/` 폴더 제외 없이 `.pt` 자동 필터링
- `environment.yml` 최신화



---
## 3. 프로젝트 소개(업데이트 중)
다양한 경구약의 위치와 종류를 인식하는 AI 시스템입니다.
모바일 스캔 기반 약물 정보 확인, 약 분류 자동화 등의 응용을 목표로 개발되었습니다.


### 🔍 프로젝트 개요
- **문제 정의**: 실제 현장에서는 환자가 복용 중인 약을 식별하기 어렵고, 잘못된 처방이 발생할 수 있음  
- **해결 방식**: 이미지 기반 알약 탐지 및 분류 모델을 통해 자동화된 식별 시스템 구축  
- **모델 선택**: 가볍고 성능이 좋은 YOLOv8을 채택하여 mAP50-95 중심의 성능 최적화  
- **기대 효과**: 약국, 병원, 보호자 등 다양한 사용자 환경에서 정확하고 빠른 약물 식별 가능


### 📁 디렉토리 구조(수정 중)
```
pill-detect-ai/
    ├── data/                  # 원본, 변환 데이터, 증강 이미지
    │   └── yolo/              # YOLO 포맷 데이터셋
    ├── src/                   # 주요 모듈 (전처리, 학습, 추론 등)
    ├── scripts/               # 실행 스크립트 (train.py, infer.py 등)
    ├── models/                # 가중치, 결과 모델 저장 폴더
    ├── outputs/               # 예측 결과 이미지, mAP, confusion matrix 등
    ├── configs/               # yaml, 하이퍼파라미터 설정 파일
    ├── environment.yml        # conda 환경 설정
    └── README.md              # 프로젝트 문서
```


### ⚙️ 설치 방법
```bash
# 가상환경 생성
conda env create -f environment/base.yml
conda activate superdogenv
```


### 🚀 실행 방법
```bash
# 학습
python scripts/train.py --config configs/train_config.yaml

# 추론 (best.pt로 예측 결과 생성)
python scripts/predict.py \
  --img data/yolo/images/test \
  --weights models/yolov10/best.pt \
  --output outputs/test_results
```


### 🧠 사용 기술

- Python 3.11, Conda
- Ultralytics YOLOv10
- PyTorch 2.6.0 + cu124
- Albumentations (bbox-aware augmentation)
- OpenCV (이미지 처리)


### 📊 모델 성능

| 실험 이름         | Epoch | Batch | mAP@0.5 | mAP@0.5:0.95 |
|------------------|--------|-------|---------|--------------|
| yolo8_32_100_1  | 100    | 32    | 000    | 000         |
| yolo9_32_100_1  | 100    | 32    | 000    | 000         |

- 📌 평가 지표: mAP@0.5, mAP@0.5:0.95 (작은 객체 성능 포함)
- 📌 73 클래스 기준 다중 객체 탐지, 오버샘플링 + 증강 병행


### 🧪 데이터셋

- 내부 커스텀 약 이미지 데이터셋 (학습 1000장, 테스트 800장)
- JSON 어노테이션 → YOLO 포맷 자동 변환
- 클래스 불균형 개선: 소수 클래스 오버샘플링, 증강 파이프라인 적용



### 👥 팀원

| 이름       | 역할         |  이메일                                        |
|------------|------------|------------------------------------------------|
| 윤승호(팀장) | 공부        | [@your-github](https://github.com/your-github) |
| 김민경      | 공부        | [@your-github](https://github.com/your-github) |
| 이현도      | 공부        | [@your-github](https://github.com/your-github) |
| 조민정      | 공부        | [@your-github](https://github.com/your-github) |
| 박창훈      | 공부        | [@your-github](https://github.com/your-github) |
| 강동우      | 공부        | [@your-github](https://github.com/your-github) |


---
## 참고 자료

- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [COCO to YOLO 변환 스크립트 참고](https://github.com/ultralytics/JSON2YOLO)



---
## 라이선스

 본 페이지를 통해 제공하는 모든 자료는 저작권법에 의해 보호받는 ㈜코드잇의 자산이며, 무단 사용 및 도용, 복제 및 배포를 금합니다. 스프린트 과정 외부로의 링크 공유 등, 모든 형태의 유출을 금합니다.

⚠️ Copyright 2025 코드잇 Inc. All rights reserved.



