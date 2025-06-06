import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
from collections import Counter
import configs.config_paths as cc

# ────────────────────────────────
# 📁 경로 설정
label_dir = os.path.join(cc.LABEL_PATH, 'train')  # 폴더명 변경 가능
save_csv_path = os.path.join(cc.PRE_OUT_DIR, 'classes_train.csv')  # 파일명 변경 가능
category_id_map_path = cc.CAT_ID_DIR
dataset_yaml_path = cc.YAML_DIR

# ────────────────────────────────
# 📥 파일 불러오기
try:
    with open(category_id_map_path, encoding='utf-8') as jf:
        yolo_to_categoryid = json.load(jf)
except FileNotFoundError:
    print(f"⚠️ JSON 매핑 파일을 찾을 수 없습니다: {category_id_map_path}")
    exit() # 파일 없으면 종료
except json.JSONDecodeError:
    print(f"⚠️ JSON 매핑 파일 형식이 잘못되었습니다: {category_id_map_path}")
    exit() # 파일 형식 오류 시 종료

try:
    with open(dataset_yaml_path, encoding='utf-8') as yf:
        data_yaml = yaml.safe_load(yf)
        if 'names' not in data_yaml:
            print(f"⚠️ YAML 파일에 'names' 키가 없습니다: {dataset_yaml_path}")
            exit()
        yolo_names = data_yaml['names']  # YOLO 클래스 ID → 실제 클래스 이름
except FileNotFoundError:
    print(f"⚠️ 데이터셋 YAML 파일을 찾을 수 없습니다: {dataset_yaml_path}")
    exit()
except yaml.YAMLError:
    print(f"⚠️ 데이터셋 YAML 파일 형식이 잘못되었습니다: {dataset_yaml_path}")
    exit()

# ────────────────────────────────
# 🔍 클래스 등장 횟수 세기
class_counter = Counter() # 각 클래스 ID별 등장 횟수를 저장할 Counter 객체

# 지정된 라벨 디렉토리가 존재하는지 확인
if not os.path.isdir(label_dir):
    print(f"⚠️ 라벨 디렉토리를 찾을 수 없습니다: {label_dir}")
    exit()

label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')] # .txt 확장자를 가진 라벨 파일 목록

if not label_files:
    print(f"⚠️ 라벨 디렉토리에 분석할 .txt 파일이 없습니다: {label_dir}")
    exit()

print(f"\n📦 '{label_dir}' 폴더의 라벨 파일 분석 중:")
# tqdm을 사용하여 진행 상황 표시
for file_name in tqdm(label_files, desc="라벨 파일 처리 중", unit="개"):
    file_path = os.path.join(label_dir, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 빈 줄이거나 공백만 있는 줄은 건너뜀
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                parts = stripped_line.split()
                if not parts: # split 결과가 비어있으면 건너뜀
                    continue
                    
                try:
                    # 라인의 첫 번째 요소가 YOLO 클래스 ID
                    cls_id = int(parts[0])
                    class_counter[cls_id] += 1
                except ValueError:
                    print(f"⚠️ 파일 '{file_name}'의 라인에서 클래스 ID를 정수로 변환하는 데 실패했습니다: {line.strip()}")
                except IndexError:
                    print(f"⚠️ 파일 '{file_name}'의 라인이 비어있거나 형식이 잘못되었습니다: {line.strip()}")

    except Exception as e:
        print(f"⚠️ 파일 '{file_name}' 처리 중 오류 발생: {e}")


# ────────────────────────────────
# 📊 결과 출력 + CSV 저장용 리스트 생성
print("\n🎯 클래스별 라벨 등장 횟수:")
csv_rows = [] # CSV 파일로 저장할 데이터를 담을 리스트

# 정렬된 클래스 ID 순서로 출력 (실제 등장한 클래스만)
for cls_id in sorted(class_counter.keys()):
    count = class_counter[cls_id]
    
    # yolo_to_categoryid는 문자열 키를 가질 수 있으므로 str(cls_id)로 조회
    category_id_str = str(cls_id)
    if category_id_str not in yolo_to_categoryid:
        print(f"⚠️ YOLO 클래스 ID '{cls_id}'에 대한 Category ID 매핑이 JSON 파일에 없습니다.")
        resolved_category_id = "매핑 없음" # 또는 다른 기본값
    else:
        resolved_category_id = yolo_to_categoryid[category_id_str]

    # yolo_names는 정수 인덱스를 사용
    if cls_id < 0 or cls_id >= len(yolo_names):
        print(f"⚠️ YOLO 클래스 ID '{cls_id}'가 YAML 파일의 'names' 리스트 범위를 벗어났습니다.")
        pill_name = "이름 없음" # 또는 다른 기본값
    else:
        pill_name = yolo_names[cls_id]

    # 콘솔 출력 (요청 포맷)
    # str(resolved_category_id)로 통일하여 어떤 타입이든 문자열로 안전하게 처리
    print(f"YOLO Class {cls_id:<2d} | Category ID {str(resolved_category_id):<5} | {count:<5}개 | {pill_name}")

    # CSV 행 저장
    csv_rows.append({
        'YOLO_class': cls_id,
        'Category_ID': resolved_category_id,
        'Pill_Name': pill_name,
        'Count': count
    })

# ────────────────────────────────
# 📁 CSV 저장
if csv_rows: # 저장할 데이터가 있을 경우에만 CSV 생성
    df = pd.DataFrame(csv_rows)
    try:
        # CSV 저장 디렉토리 확인 및 생성
        save_dir = os.path.dirname(save_csv_path)
        if save_dir and not os.path.exists(save_dir): # save_dir이 비어있지 않고 존재하지 않으면
            os.makedirs(save_dir)
            
        df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 클래스 통계 CSV 저장 완료: {save_csv_path}")
    except IOError as e:
        print(f"\n⚠️ CSV 파일 저장 중 오류 발생: {e}")
    except Exception as e:
        print(f"\n⚠️ CSV 파일 저장 중 예기치 않은 오류 발생: {e}")
else:
    print("\n분석된 클래스 데이터가 없어 CSV 파일을 저장하지 않습니다.")