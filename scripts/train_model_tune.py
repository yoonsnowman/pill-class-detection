from ultralytics import YOLO
import torch
import os
import pandas as pd
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 설정
# 모델 가중치 저장 경로(없으면 새로 자동 다운로드함)
pt_dir = 'data/yolo/yolov9e.pt' 

# yaml 파일 경로
yaml_dir = 'data/yolo/pill_yolo_format/data.yaml'

# 출력 폴더 경로
out_folder_dir = 'data/yolo/run'

# 출력 폴더 이름
out_folder_name = '9e_run4' # 수정하여 실행 이름 지정

# 테스트 이미지 경로
test_image_dir = 'data/yolo/pill_yolo_format/images/test/'

# 캐글 제출 csv파일 저장 경로
submission_csv_path = f'{out_folder_dir}/{out_folder_name}/submission.csv'

# 추론 시 사용할 모델 경로
trained_model_path  = f'{out_folder_dir}/{out_folder_name}/weights/best.pt'

# categoryid 저장된 파일 경로
categoryid_json_path = 'data/yolo/pill_yolo_format/yolo_to_categoryid.json'

# 나눔고딕 폰트 경로 지정
nanum_font_path = 'data/font/NanumGothic.ttf'


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if os.path.exists(nanum_font_path):
    fm.fontManager.addfont(nanum_font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
else:
    print(f"⚠️ 나눔고딕 폰트를 찾을 수 없습니다: {nanum_font_path}")



def train_yolo():    

    hyp_aug = dict(
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=2.0,
        translate=0.15,
        scale=0.6,
        shear=0.05,
        perspective=0.0002,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=50,
        weight_decay=0.0005
    )

    print("\n✅ 학습 시작")

    model = YOLO(pt_dir)
    model.train(
        data=yaml_dir,
        epochs=100,
        imgsz=640,
        batch=16,
        project=out_folder_dir,
        name=f'{out_folder_name}',
        device=device,
        exist_ok=True,
        **hyp_aug
    )

    print(f"\n✅ 전체 학습 완료! 결과: {out_folder_dir}/{out_folder_name}/weights/best.pt")







def tune_conf_iou(model_to_tune_path, data_yaml, imgsz_val=640, batch_val=16, metric_to_optimize='metrics/mAP50(B)'):
    """
    학습된 모델을 사용하여 최적의 conf 및 iou 값을 튜닝합니다.
    Args:
        model_to_tune_path (str): 튜닝할 학습된 모델(.pt)의 경로.
        data_yaml (str): data.yaml 파일 경로 (검증셋 경로 포함).
        imgsz_val (int): 검증 시 이미지 크기.
        batch_val (int): 검증 시 배치 크기.
        metric_to_optimize (str): 최적화할 평가지표 키 (예: 'metrics/mAP50(B)', 'metrics/mAP50-95(B)').
                                   `model.val()` 결과 객체의 `keys()`를 확인하여 정확한 키를 사용하세요.
    """
    if not os.path.exists(model_to_tune_path):
        print(f"⚠️ 튜닝할 모델 파일을 찾을 수 없습니다: {model_to_tune_path}")
        print("먼저 모델을 학습시켜주세요 ('--mode train' 또는 '--mode both').")
        return None, None, -1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_to_tune_path)
    print(f"[TUNE] 모델 로드 완료: {model_to_tune_path}")

    # 탐색할 conf 및 iou 값 범위 설정
    # 필요에 따라 범위와 간격을 더 세밀하게 또는 넓게 조정하세요.
    conf_values = np.arange(0.1, 0.51, 0.05).tolist()  # 예: [0.1, 0.15, ..., 0.5]
    iou_values = np.arange(0.3, 0.61, 0.05).tolist()    # 예: [0.3, 0.35, ..., 0.6]
    # conf_values = [0.25] # 빠른 테스트용
    # iou_values = [0.45]  # 빠른 테스트용


    print(f"[TUNE] Conf 값 탐색 범위: {conf_values}")
    print(f"[TUNE] IoU 값 탐색 범위: {iou_values}")
    print(f"[TUNE] 최적화 대상 지표: {metric_to_optimize}")

    best_metric_val = -1.0
    best_conf = -1.0
    best_iou = -1.0
    
    results_log = []

    for conf_val in conf_values:
        for iou_val in iou_values:
            conf_val = round(conf_val, 2) # 소수점 정리
            iou_val = round(iou_val, 2)   # 소수점 정리
            print(f"\n[TUNE] 검증 시작: conf={conf_val}, iou={iou_val}")
            try:
                metrics = model.val(
                    data=data_yaml,
                    imgsz=imgsz_val,
                    batch=batch_val,
                    conf=conf_val,
                    iou=iou_val,
                    split='val', # 'val' 또는 'test' (data.yaml에 정의된 대로)
                    save_json=False, # COCO mAP 계산용 JSON 저장 안 함
                    save_hybrid=False, # Hybrid 형식 레이블 저장 안 함
                    device=device,
                    verbose=False # 너무 많은 로그 출력을 방지
                )
                
                # metrics 객체에서 원하는 평가지표 추출
                # metrics.box.map, metrics.box.map50, metrics.box.map75, metrics.box.map50_95
                # 사용 가능한 키는 metrics.keys() 또는 metrics.box.keys()로 확인 가능
                current_metric_val = 0.0
                if metric_to_optimize == 'metrics/mAP50(B)' and hasattr(metrics.box, 'map50'):
                    current_metric_val = metrics.box.map50
                elif metric_to_optimize == 'metrics/mAP50-95(B)' and hasattr(metrics.box, 'map'):
                    current_metric_val = metrics.box.map # map is mAP50-95
                else:
                    print(f"⚠️ '{metric_to_optimize}' 지표를 찾을 수 없습니다. 사용 가능한 box 지표: {metrics.box.keys()}")
                    # 기본적으로 mAP50을 사용하거나, 사용자가 직접 수정하도록 안내
                    if hasattr(metrics.box, 'map50'):
                        print("기본으로 'metrics/mAP50(B)' (metrics.box.map50)을 사용합니다.")
                        current_metric_val = metrics.box.map50
                    else:
                        print("적절한 평가지표를 찾을 수 없어 이 조합은 건너<0xEB><08><0x81>니다.")
                        continue

                print(f"[TUNE] 결과: conf={conf_val}, iou={iou_val}, {metric_to_optimize}={current_metric_val:.4f}")
                results_log.append({'conf': conf_val, 'iou': iou_val, metric_to_optimize: current_metric_val})

                if current_metric_val > best_metric_val:
                    best_metric_val = current_metric_val
                    best_conf = conf_val
                    best_iou = iou_val
                    print(f"⭐ [TUNE] 새로운 최적값 발견! conf={best_conf}, iou={best_iou}, {metric_to_optimize}={best_metric_val:.4f}")

            except Exception as e:
                print(f"⚠️ [TUNE] conf={conf_val}, iou={iou_val} 검증 중 오류 발생: {e}")
                continue
    
    print("\n--- [TUNE] 튜닝 결과 요약 ---")
    for log in sorted(results_log, key=lambda x: x[metric_to_optimize], reverse=True):
        print(f"Conf: {log['conf']:.2f}, IoU: {log['iou']:.2f}, {metric_to_optimize}: {log[metric_to_optimize]:.4f}")

    if best_conf != -1:
        print(f"\n🏆 [TUNE] 최종 최적 조합: conf={best_conf:.2f}, iou={best_iou:.2f} (이때 {metric_to_optimize} = {best_metric_val:.4f})")
        return best_conf, best_iou, best_metric_val
    else:
        print("😔 [TUNE] 최적의 조합을 찾지 못했습니다. 탐색 범위나 설정을 확인해주세요.")
        return None, None, -1




def predict_and_generate_submission(pred_conf=0.25, pred_iou=0.45, use_tta=True):
    print(f"[2] 모델 로드 및 추론 시작 (conf={pred_conf}, iou={pred_iou}, TTA={use_tta})...")
    
    if not os.path.exists(trained_model_path):
        print(f"⚠️ 예측할 모델 파일을 찾을 수 없습니다: {trained_model_path}")
        print("먼저 모델을 학습시켜주세요 ('--mode train' 또는 '--mode both').")
        return

    model = YOLO(trained_model_path)

    results = model.predict(
        source=test_image_dir,
        imgsz=640,
        conf=pred_conf, # 튜닝된 또는 지정된 conf 값 사용
        iou=pred_iou,   # 튜닝된 또는 지정된 iou 값 사용
        augment=use_tta, # TTA 사용 여부
        save=False,
        stream=True
    )

    submission_rows = []
    annotation_id = 1

    if not os.path.exists(categoryid_json_path):
        print(f"⚠️ Category ID JSON 파일을 찾을 수 없습니다: {categoryid_json_path}")
        return
        
    with open(categoryid_json_path, 'r', encoding='utf-8') as f:
        yolo_to_dl_idx = {int(k): int(v) for k, v in json.load(f).items()}

    for r_idx, r in enumerate(results): # enumerate for progress tracking
        if (r_idx + 1) % 100 == 0:
             print(f"  [PREDICT] 이미지 {r_idx + 1}개 처리 중...")
        file_name = os.path.basename(r.path)
        match = re.search(r'(\d+)\.\w+', file_name) # 파일 확장자 앞의 숫자만 추출
        if not match:
            print(f"⚠️ 이미지 파일명에서 숫자 추출 실패: {file_name}")
            continue
        image_id = int(match.group(1))

        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            score = float(box.conf[0])
            
            yolo_cls_id = int(box.cls[0])
            if yolo_cls_id not in yolo_to_dl_idx:
                print(f"⚠️ yolo_to_dl_idx에 없는 YOLO 클래스 ID: {yolo_cls_id} (이미지: {file_name})")
                continue
            category_id = yolo_to_dl_idx[yolo_cls_id]

            submission_rows.append([
                annotation_id, image_id, category_id,
                int(x1), int(y1), int(w), int(h), round(score, 4)
            ])
            annotation_id += 1
    
    if not submission_rows:
        print("⚠️ 제출할 예측 결과가 없습니다. 모델 성능 또는 데이터, 설정을 확인해주세요.")
        return

    columns = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
    df = pd.DataFrame(submission_rows, columns=columns)
    
    df.to_csv(submission_csv_path, index=False, encoding='utf-8-sig')
    print(f"[3] ✅ 제출용 CSV 저장 완료: {submission_csv_path} (총 {len(df)}개 라벨)")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 모델 학습, 튜닝 및 예측 스크립트")
    parser.add_argument('--mode', choices=['train', 'tune', 'predict', 'both', 'train_tune_predict'], default='both',
                        help="실행 모드: \n"
                             "'train': 모델 학습만 수행.\n"
                             "'tune': 학습된 모델로 conf/iou 튜닝 수행.\n"
                             "'predict': 학습된 모델로 테스트셋 예측 및 제출 파일 생성.\n"
                             "'both': 'train' 후 'predict' 수행 (기본 conf/iou 사용).\n"
                             "'train_tune_predict': 'train', 'tune', 'predict' 순차 수행 (튜닝된 conf/iou 사용).")
    parser.add_argument('--conf', type=float, default=0.25, help="예측 시 사용할 신뢰도 임계값 (predict 모드에서만 직접 사용)")
    parser.add_argument('--iou', type=float, default=0.45, help="예측 시 사용할 NMS IoU 임계값 (predict 모드에서만 직접 사용)")
    parser.add_argument('--tta', action='store_true', help="예측 시 Test-Time Augmentation 사용 여부 (predict 모드)")
    parser.add_argument('--no_tta', action='store_false', dest='tta', help="예측 시 Test-Time Augmentation 사용 안 함 (predict 모드, 기본값)")
    parser.set_defaults(tta=True) # TTA 기본값 True로 설정

    args = parser.parse_args()


    if args.mode == 'train':
        train_yolo()
    elif args.mode == 'tune':
        print("--- Conf/IoU 튜닝 시작 ---")
        # 튜닝할 모델 경로, data.yaml 경로, 검증 이미지 크기, 검증 배치 크기 등을 전달
        # metric_to_optimize는 캐글 대회 평가 지표에 맞춰 'metrics/mAP50(B)' 또는 'metrics/mAP50-95(B)' 등으로 설정
        best_conf, best_iou, best_metric = tune_conf_iou(
            model_to_tune_path=trained_model_path,
            data_yaml=yaml_dir,
            imgsz_val=640,
            batch_val=16, # GPU 메모리에 맞춰 조정
            metric_to_optimize='metrics/mAP50-95(B)' # 예시: mAP@0.5를 최적화
        )
        if best_conf is not None:
            print(f"\n튜닝 완료. 최적 conf: {best_conf}, 최적 iou: {best_iou} (이때 metric: {best_metric:.4f})")
            print(f"이 값을 사용하여 '--mode predict --conf {best_conf} --iou {best_iou}' 로 예측을 실행할 수 있습니다.")
        else:
            print("\n튜닝에 실패했거나 적절한 값을 찾지 못했습니다.")

    elif args.mode == 'predict':
        predict_and_generate_submission(pred_conf=args.conf, pred_iou=args.iou, use_tta=args.tta)
    elif args.mode == 'both':
        train_yolo()
        print("\n--- 학습 완료. 기본값으로 예측 시작 ---")
        predict_and_generate_submission(pred_conf=args.conf, pred_iou=args.iou, use_tta=args.tta) # 기본값 또는 인자로 받은 값 사용
    elif args.mode == 'train_tune_predict':
        train_yolo()
        print("\n--- 학습 완료. Conf/IoU 튜닝 시작 ---")
        best_conf_tuned, best_iou_tuned, best_metric_tuned = tune_conf_iou(
            model_to_tune_path=trained_model_path,
            data_yaml=yaml_dir,
            imgsz_val=640,
            batch_val=16,
            metric_to_optimize='metrics/mAP50(B)' # 캐글 평가지표에 맞게 수정하세요 (예: 'metrics/mAP50-95(B)')
        )
        if best_conf_tuned is not None and best_iou_tuned is not None:
            print(f"\n--- 튜닝 완료. 최적값(conf={best_conf_tuned}, iou={best_iou_tuned})으로 예측 시작 ---")
            predict_and_generate_submission(pred_conf=best_conf_tuned, pred_iou=best_iou_tuned, use_tta=args.tta)
        else:
            print("\n--- 튜닝 실패. 기본값으로 예측 시작 ---")
            predict_and_generate_submission(pred_conf=args.conf, pred_iou=args.iou, use_tta=args.tta)



# 터미널 사용 예시:
# 1. 학습만:
#    python your_script_name.py --mode train
#
# 2. 학습된 모델로 conf/iou 튜닝만:
#    python your_script_name.py --mode tune
#    (out_folder_name이 학습 시와 동일해야 trained_model_path가 올바르게 설정됩니다)
#
# 3. 학습된 모델로 예측만 (기본 또는 지정된 conf/iou 사용):
#    python your_script_name.py --mode predict
#    python your_script_name.py --mode predict --conf 0.3 --iou 0.5 --no_tta
#
# 4. 학습 후 기본 conf/iou로 예측:
#    python your_script_name.py --mode both
#
# 5. 학습 -> conf/iou 튜닝 -> 튜닝된 값으로 예측:
#    python your_script_name.py --mode train_tune_predict
