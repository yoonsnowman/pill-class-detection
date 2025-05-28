# config/config_dirs.py

####################################################
# 내용: 코랩 연동 directory 목록
# 작성자: 윤승호
# 수정일: 2025. 05. 28. 12:00
# 용도: 로컬 경로와 구글 드라이브 경로 연동
####################################################


# train_dir = DIR['raw_train'] 이런 식으로 호출해서 사용
def get_dirs(environment='base'):
    return {
        'environment': f'environment/{environment}',
        'raw_train':   'data/raw/train_images',
        'raw_val':     'data/raw/val_images',
        'raw_test':    'data/raw/test_images',
        'raw_labels':  'data/raw/processed/labels',
        'raw_classes': 'data/raw/processed/classes.txt',
        'font':        'data/font/NanumGothic.ttf',
        'trainset':    'data/train',
        'valset':      'data/val',
        'testset':     'data/test',
    }