import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2

import numpy as np
from ultralytics import YOLO

from work_with_imgs import work_with_imgs
from extract_from_video import extract_frames_opencv

def check_model(path: str = 'best.pt'):
    model = YOLO(path)

    # 1. Проверка на test
    test_metrics = model.val(data='dataset.yaml', split='test')
    print(f"mAP50 test: {test_metrics.box.map50:.3f}")

    # 2. Проверка на train (должно быть ~1.0)
    train_metrics = model.val(data='dataset.yaml', split='train')
    print(f"mAP50 train: {train_metrics.box.map50:.3f}")

    # 3. Разница между train и test
    gap = train_metrics.box.map50 - test_metrics.box.map50
    if gap > 0.1:
        print("ПЕРЕОБУЧЕНИЕ! Разница > 0.1")
    elif gap > 0.05:
        print("Лёгкое переобучение")
    else:
        print("Отличный результат, нет переобучения")


if __name__ == '__main__':

    mode = input('all/train/check ')
    if mode == 'check':
        check_model()
    else:
        if mode == 'all':
            extract_frames_opencv(video_path='vids/vid01', output_folder='frames_output', frame_interval=10)
            work_with_imgs(imgs_folder='test_imgs', annots_folder='test_annots', augments=5)
            
        model = YOLO('yolo11s.pt')



        results = model.train(
            data='dataset.yaml',   
            optimizer='AdamW',  
            project='fruit_detector',
            name='exp1', 
            warmup_epochs=3,
            warmup_bias_lr=0.1,
            epochs=100,              
            batch=16,                 
            imgsz=640,
            lr0=0.001,               
            patience=20,                    
            workers=8,
            plots=True,
            cos_lr=True,  
            augment=False
        )