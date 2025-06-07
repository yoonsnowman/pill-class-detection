**Project Version 1.4.1**

---
>data/yolo/ 경로에 프로젝트 압축 파일 해제 후 사용하시기 바랍니다.
>train_images, train_annotations, test_images 폴더가 존재해야 합니다.

## 1. 명령어 사용 방법
### main.py
기본 셋팅값으로 전처리, 오버샘플링, 클래스 통계, 모델 학습, 추론 순차 실행
- `python -m main`


### configs/config_paths.py
- `import configs.config_paths as cc` 스크립트에 추가
- 스크립트에서 `cc.PRE_IN_DIR` 등을 사용하여 경로 호출


### src/train.py
수동 셋팅값 사용(모델pt명, outputs/에 저장할 폴더명, 배치, 에포크)
- `python -m src.train --model yolov8n --name myfoldername --b 16 --e 10`


### src/preprocess.py
명령어 특이사항 없음
- `python -m src.preprocess`


### src/predict.py
output/에 저장할 폴더명 명령어에 지정
- `python -m src.predict --name myfoldername`


### scripts/class_counter.py
기본 탐색 폴더: data/yolo/labels/train/ & data/yolo/labels/train_aug/
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
## 2. 패치노트
### 변경사항
- `main.py`로 전처리, 오버샘플링, 클래스 통계, 모델 학습, 추론 한번에 실행 가능하도록 변경
- `notebooks/`에 EDA 노트북 추가
- `models/` 폴더 내 모델별 간략 정보 `.txt`로 정리


### 적용완료
- 모든 경로를 `config_paths.py`에 통합 → `import configs.config_paths as cc` 형태로 호출
- `data/detr/` 디렉토리 제거
- 모든 학습 결과 → `outputs/` 폴더로 통합 관리
- `.gitignore` 업데이트:
    - `weights/` 폴더 제외 없이 `.pt` 자동 필터링
- `environment.yml` 최신화



---
## 3. 프로젝트 소개
다양한 경구약의 위치와 종류를 인식하는 AI 시스템입니다.
모바일 스캔 기반 약물 정보 확인, 약 분류 자동화 등의 응용을 목표로 개발되었습니다.


### 🔍 프로젝트 개요
- **문제 정의**: 환자가 복용 중인 약을 스스로 식별하기 어렵고, 잘못된 처방이 발생할 수 있음  
- **해결 방식**: 이미지 기반 알약 탐지 및 분류 모델을 통해 자동화된 식별 시스템 구축  
- **모델 선택**: 가볍고 성능이 좋은 YOLOv8을 채택하여 mAP50 중심의 성능 최적화  
- **기대 효과**: 약국, 병원, 일반 사용자, 알약 제조공장 등 다양한 환경에서 정확하고 빠른 식별 가능


### ⚙️ 설치 방법
```bash
# 가상환경 생성
conda env create -f environment/base.yml
conda activate superdogenv
```


### 🧪 데이터셋

- 캐글 경진대회 이미지 데이터셋 (학습 1489장, 테스트 843장)
- COCO 포맷 JSON 어노테이션 → YOLO 포맷 TXT 어노테이션 변환
- 클래스 불균형 개선: 소수 클래스 오버샘플링, 증강 파이프라인 적용


### 👥 팀원

| 이름        | 역할        |
|------------|------------|
| 윤승호      | 팀장        |
| 김민경      | 팀원        |
| 이현도      | 팀원        |
| 조민정      | 팀원        |
| 박창훈      | 팀원        |
| 강동우      | 팀원        |


### 참고 자료

- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)


### 라이선스

 본 페이지를 통해 제공하는 모든 자료는 저작권법에 의해 보호받는 ㈜코드잇의 자산이며, 무단 사용 및 도용, 복제 및 배포를 금합니다. 스프린트 과정 외부로의 링크 공유 등, 모든 형태의 유출을 금합니다.

⚠️ Copyright 2025 코드잇 Inc. All rights reserved.



