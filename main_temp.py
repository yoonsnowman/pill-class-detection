#####################
# 만드는 중입니다
#####################


from src.train import train_yolo_model

def main():
    # --- 하이퍼파라미터 수동 설정 ---
    model_name = 'yolov8n'
    run_folder = 'v8n_32_50'
    batch = 32
    epochs = 50

    # --- 경로 조합 ---
    model_path = f'models/{model_name}.pt'

    # --- 학습 함수 실행 ---
    train_yolo_model(
        base_model_actual_path=model_path,
        run_folder_name=run_folder,
        batch_size=batch,
        epochs=epochs
    )

if __name__ == "__main__":
    main()
