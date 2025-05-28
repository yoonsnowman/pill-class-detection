import os
import shutil
from tqdm import tqdm

def organize_annotations_for_val_with_prefix_in_folder_name_case_sensitive(
    original_annotations_source_dir, # 원본 주석 폴더들이 모두 들어있는 폴더 (data/train_annotations)
    val_images_dir,                  # 검증 이미지가 있는 폴더 (data/val_images)
    val_annotations_output_dir       # 검증 주석이 이동할 새로운 폴더 (data/val_annotations)
):
    """
    val_images 디렉토리의 이미지 파일명에서 '_' 앞부분을 추출하여,
    original_annotations_source_dir 내의 최상위 주석 폴더 이름에 해당 접두사가 '포함'되는 경우,
    해당 주석 폴더를 val_annotations_output_dir 디렉토리로 이동합니다.
    대소문자를 구분합니다.
    """
    if not os.path.exists(val_annotations_output_dir):
        os.makedirs(val_annotations_output_dir)
        print(f"✅ 검증 주석 디렉토리 생성: {val_annotations_output_dir}")

    # 1. val_images 디렉토리에서 이미지 파일의 '첫 번째 접두사' (언더바 앞부분) 추출
    # 이 접두사가 주석 폴더 이름에 포함될지 확인하는 기준이 됩니다.
    val_image_prefixes = set()
    print(f"\n{val_images_dir}에서 검증 이미지 접두사 추출 중...")
    for img_filename in os.listdir(val_images_dir):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 파일명에서 확장자를 제거하고, '_'를 기준으로 분리하여 첫 번째 부분만 가져옵니다.
            # .lower()를 제거하여 대소문자를 그대로 유지합니다.
            prefix = os.path.splitext(img_filename)[0].split('_')[0].strip()
            val_image_prefixes.add(prefix)
    print(f"➡️ 총 {len(val_image_prefixes)}개의 검증 이미지 접두사 추출 완료.")
    print(f"   추출된 접두사 예시 (최대 10개): {list(val_image_prefixes)[:10]}") # 첫 10개 출력

    print(f"\n{original_annotations_source_dir}에서 주석 폴더 이동 시도 중 (이미지 접두사 포함 여부 매칭, 대소문자 구분)...")

    moved_count = 0
    # 이미 이동된 최상위 폴더의 이름을 기록하여 중복 이동 방지
    already_moved_top_level_folders = set() 

    # original_annotations_source_dir 바로 아래에 있는 폴더들만 확인
    # 이 폴더들이 'K-알약코드-알약코드-...' 또는 'K-알약코드-알약코드-..._json' 형태일 수 있습니다.
    top_level_annotation_folders = [d for d in os.listdir(original_annotations_source_dir) 
                                    if os.path.isdir(os.path.join(original_annotations_source_dir, d))]

    if not top_level_annotation_folders:
        print(f"⚠️ 경고: {original_annotations_source_dir} 디렉토리 내에 하위 폴더가 없습니다. 경로를 확인해주세요.")
        return # 하위 폴더가 없으면 더 이상 진행할 필요 없음

    for top_level_folder_name_raw in tqdm(top_level_annotation_folders, desc="최상위 주석 폴더 처리 중"):
        current_source_path = os.path.join(original_annotations_source_dir, top_level_folder_name_raw)

        # 이미 처리된 폴더는 건너뛰기
        if top_level_folder_name_raw in already_moved_top_level_folders:
            continue

        # 주석 폴더 이름을 그대로 사용 (대소문자 구분)하고 양쪽 공백 제거
        top_level_folder_name_to_match = top_level_folder_name_raw.strip()

        is_val_annotation_folder = False
        # 추출된 이미지 접두사 중 하나라도 현재 주석 폴더 이름에 포함되는지 확인
        # 예: 'K-001900-010224-016551-031705' (이미지 접두사)가 'K-001900-010224-016551-031705' 또는 'K-001900-010224-016551-031705_json' 안에 있는지 확인
        for img_prefix in val_image_prefixes:
            if img_prefix in top_level_folder_name_to_match:
                is_val_annotation_folder = True
                # print(f"   ✨ 매칭 성공! 이미지 접두사 '{img_prefix}'이(가) 주석 폴더 '{top_level_folder_name_to_match}'에 포함됩니다.") # 디버그용
                break
        
        if is_val_annotation_folder:
            destination_path = os.path.join(val_annotations_output_dir, top_level_folder_name_raw)
            
            try:
                print(f"➡️ 이동 시도: '{current_source_path}' -> '{destination_path}'") # 이동 시도 시 출력
                shutil.move(current_source_path, destination_path)
                moved_count += 1
                already_moved_top_level_folders.add(top_level_folder_name_raw) # 이동된 폴더 기록
                print(f"✅ 이동 완료: '{top_level_folder_name_raw}'")
            except Exception as e:
                print(f"❌ 이동 오류 발생 ({current_source_path}): {e}")
                import traceback
                traceback.print_exc() # 상세한 에러 스택 트레이스 출력
        # else:
            # print(f"  ❌ 매칭 실패: '{top_level_folder_name_to_match}'은(는 어떤 검증 이미지 접두사도 포함하지 않습니다.") # 디버그용
    
    print(f"\n✅ 주석 폴더 정리 완료. 총 {moved_count}개의 폴더가 {val_annotations_output_dir}로 이동되었습니다.")
    print(f"남아있는 주석 폴더들은 {original_annotations_source_dir}에 유지됩니다.")

# --- 사용 예시 ---
if __name__ == "__main__":
    # 데이터셋 경로 설정
    original_annotations_folder = 'data/train_annotations' # 원본 주석들이 모두 있는 폴더
    val_images_folder = 'data/val_images'                 # 1단계에서 분리된 검증 이미지가 있는 폴더
    val_annotations_output_folder = 'data/val_annotations' # 검증 주석이 이동할 새로운 폴더

    # 스크립트 실행: 이미지 분리 작업 후에 이 스크립트를 실행합니다.
    organize_annotations_for_val_with_prefix_in_folder_name_case_sensitive(
        original_annotations_source_dir=original_annotations_folder,
        val_images_dir=val_images_folder,
        val_annotations_output_dir=val_annotations_output_folder
    )