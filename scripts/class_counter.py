import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
from collections import Counter
import configs.config_paths as cc

def process_labels(subfolder: str):
    """
    지정한 라벨 하위 폴더(예: 'train' 또는 'train_aug')를 순회하며
    클래스별 등장 횟수를 세고, CSV로 저장합니다.
    """
    # ────────────────────────────────
    # 📁 경로 설정
    label_dir = os.path.join(cc.LABEL_PATH, subfolder)  # 'train' 또는 'train_aug'
    save_csv_path = os.path.join(cc.PRE_OUT_DIR, f'classes_{subfolder}.csv')

    # ────────────────────────────────
    # 📥 JSON, YAML 파일 불러오기 (공통)
    try:
        with open(cc.CAT_ID_DIR, encoding='utf-8') as jf:
            yolo_to_categoryid = json.load(jf)
    except FileNotFoundError:
        print(f"⚠️ JSON 매핑 파일을 찾을 수 없습니다: {cc.CAT_ID_DIR}")
        return
    except json.JSONDecodeError:
        print(f"⚠️ JSON 매핑 파일 형식이 잘못되었습니다: {cc.CAT_ID_DIR}")
        return

    try:
        with open(cc.YAML_DIR, encoding='utf-8') as yf:
            data_yaml = yaml.safe_load(yf)
            if 'names' not in data_yaml:
                print(f"⚠️ YAML 파일에 'names' 키가 없습니다: {cc.YAML_DIR}")
                return
            yolo_names = data_yaml['names']
    except FileNotFoundError:
        print(f"⚠️ 데이터셋 YAML 파일을 찾을 수 없습니다: {cc.YAML_DIR}")
        return
    except yaml.YAMLError:
        print(f"⚠️ 데이터셋 YAML 파일 형식이 잘못되었습니다: {cc.YAML_DIR}")
        return

    # ────────────────────────────────
    # 🔍 클래스 등장 횟수 세기
    class_counter = Counter()

    if not os.path.isdir(label_dir):
        print(f"⚠️ 라벨 디렉토리를 찾을 수 없습니다: {label_dir}")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not label_files:
        print(f"⚠️ 라벨 디렉토리에 분석할 .txt 파일이 없습니다: {label_dir}")
        return

    print(f"[INFO]'{label_dir}' 폴더의 라벨 파일 분석")
    for file_name in tqdm(label_files, desc="[INFO] 라벨 파일 처리", unit="개"):
        file_path = os.path.join(label_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    parts = stripped_line.split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                        class_counter[cls_id] += 1
                    except ValueError:
                        print(f"⚠️ 파일 '{file_name}'의 라인에서 클래스 ID를 정수로 변환 실패: {line.strip()}")
                    except IndexError:
                        print(f"⚠️ 파일 '{file_name}'의 라인이 비어있거나 형식 오류: {line.strip()}")
        except Exception as e:
            print(f"⚠️ 파일 '{file_name}' 처리 중 오류 발생: {e}")

    # ────────────────────────────────
    # 📊 결과 출력 + CSV 저장용 리스트 생성
    csv_rows = []
    for cls_id in sorted(class_counter.keys()):
        count = class_counter[cls_id]
        category_id_str = str(cls_id)
        if category_id_str not in yolo_to_categoryid:
            print(f"⚠️ YOLO 클래스 ID '{cls_id}'에 대한 Category ID 매핑 없음.")
            resolved_category_id = "매핑 없음"
        else:
            resolved_category_id = yolo_to_categoryid[category_id_str]

        if cls_id < 0 or cls_id >= len(yolo_names):
            print(f"⚠️ YOLO 클래스 ID '{cls_id}'가 YAML 'names' 범위 벗어남.")
            pill_name = "이름 없음"
        else:
            pill_name = yolo_names[cls_id]

        csv_rows.append({
            'YOLO_class': cls_id,
            'Category_ID': resolved_category_id,
            'Pill_Name': pill_name,
            'Count': count
        })

    # ────────────────────────────────
    # 📁 CSV 저장
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        try:
            save_dir = os.path.dirname(save_csv_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ '{subfolder}' 클래스 통계 CSV 저장 완료: {save_csv_path}")
        except IOError as e:
            print(f"\n⚠️ CSV 파일 저장 중 오류 발생: {e}")
        except Exception as e:
            print(f"\n⚠️ CSV 파일 저장 중 예기치 않은 오류 발생: {e}")
    else:
        print(f"\n분석된 '{subfolder}' 데이터가 없어 CSV 파일을 저장하지 않습니다.")


if __name__ == "__main__":
    # 'train'과 'train_aug' 두 폴더를 순차적으로 처리
    for folder in ['train', 'train_aug']:
        process_labels(folder)
