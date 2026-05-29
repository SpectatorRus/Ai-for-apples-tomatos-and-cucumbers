import cv2
import os
from pathlib import Path
def extract_frames_opencv(video_path, output_folder, frame_interval=1):
    """
    Извлекает кадры из видео
    
    Args:
        video_path: путь к видеофайлу
        output_folder: папка для сохранения кадров
        frame_interval: интервал между сохраняемыми кадрами (1 = каждый кадр)
    """
    # Создаем папку для кадров, если её нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Сохраняем каждый frame_interval-й кадр
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Сохранено кадров: {saved_count}")
        
        frame_count += 1
    
    cap.release()
    print(f"Готово! Сохранено {saved_count} кадров из {frame_count}")


def rename_all_frames_in_folder(folder: str, flag: str) -> str:
    if not os.path.exists(folder):
        return 'No folder'

    folder_path = Path(folder)
    for subdir in folder_path.iterdir():
        if flag in subdir.name and subdir.is_dir():
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    # Keep file in same subdir, just rename
                    new_path = subdir / f"{subdir.name.split('_')[0]}_{file_path.name.split('_')[-2]}_{file_path.name.split('_')[-1]}"
                    file_path.rename(new_path)
    
    return 'OK'

def rewrite_yolo_class(folder: str, flag: str, class_id: int) -> str:
    if not os.path.exists(folder):
        return 'No folder'

    folder_path = Path(folder)
    for subdir in folder_path.iterdir():
        if flag in subdir.name and subdir.is_dir():
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    with open(file_path, 'r+') as file:
                        lines = file.readlines()
                        file.seek(0)
                        for line in lines:
                            content = line.strip().split()
                            content[0] = str(class_id)
                            file.write(' '.join(content) + '\n')
                        
                        file.truncate()
    
    return 'OK'
