import os, sys, yaml

def make_data_yaml():
    # Colab 여부 판단
    def is_colab():
        return 'google.colab' in sys.modules or os.path.exists('/content/drive')

    # 클래스 이름 로드
    classes_txt = 'data/yolo/raw_split/classes.txt'
    with open(classes_txt, 'r', encoding='utf-8') as f:
        names = [line.strip().split(': ')[-1] for line in f if ': ' in line]

    yaml_dict = {
        'train' : 'data/yolo/raw_split/train/images',
        'val'   : 'data/yolo/raw_split/val/images',
        'test'  : 'data/yolo/raw_split/test/images',
        'nc'    : len(names),
        'names' : names
    }

    yaml_path = 'data/yolo/raw_split/data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)

    print(f"📄 .yaml 저장 완료: {yaml_path}")


if __name__ == '__main__':
    make_data_yaml()
