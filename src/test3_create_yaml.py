import os, yaml
from autoconfig import DIR

def make_data_yaml():
    # 클래스 텍스트 파일 경로
    classes_txt = DIR('data/yolo/raw_split/classes.txt')

    # 클래스 이름 로드
    with open(classes_txt, 'r', encoding='utf-8') as f:
        names = [line.strip().split(': ')[-1] for line in f if ': ' in line]

    # DIR로 yaml 내부 경로 생성
    yaml_dict = {
        'train' : DIR('data/yolo/raw_split/train/images'),
        'val'   : DIR('data/yolo/raw_split/val/images'),
        'test'  : DIR('data/yolo/raw_split/test/images'),
        'nc'    : len(names),
        'names' : names
    }

    # 저장 경로도 autoconfig로
    yaml_path = DIR('data/yolo/raw_split/data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)

    print(f"📄 .yaml 저장 완료: {yaml_path}")


if __name__ == '__main__':
    make_data_yaml()
