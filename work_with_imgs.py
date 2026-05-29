import os
import cv2
import glob
import albumentations as A
from sklearn.model_selection import train_test_split
import yaml
import random
import numpy as np


def work_with_imgs(imgs_folder: str = 'test_imgs', annots_folder: str = 'test_annots', augments: int = 5):
    # Фиксируем seed для воспроизводимости
    random.seed(42)
    np.random.seed(42)

    # Настройки аугментации
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ToGray(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True, min_visibility=0.5))

    # Папки
    out_dirs = {
        'train': ('augmented/train/images', 'augmented/train/labels'),
        'val':   ('augmented/val/images',   'augmented/val/labels'),
        'test':  ('augmented/test/images',  'augmented/test/labels')
    }

    for img_dir, lbl_dir in out_dirs.values():
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    # 1. Собираем все имена файлов (без расширения) и разбиваем на train/val/test
    all_names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{imgs_folder}/*.jpg")]
    if not all_names:
        print("Не найдено изображений в", imgs_folder)
        exit()

    train_names, temp = train_test_split(all_names, test_size=0.3, random_state=42)
    val_names, test_names = train_test_split(temp, test_size=0.5, random_state=42)

    # Словарь: имя -> сплит
    split_map = {}
    for name in train_names:
        split_map[name] = 'train'
    for name in val_names:
        split_map[name] = 'val'
    for name in test_names:
        split_map[name] = 'test'

    print(f"Разбиение: train={len(train_names)}, val={len(val_names)}, test={len(test_names)}")

    # Функция сохранения изображения и лейблов
    def save_sample(image, bboxes, classes, split, base_name, suffix=''):
        """Сохраняет изображение и .txt лейбл в папку нужного сплита."""
        img_dir, lbl_dir = out_dirs[split]
        img_filename = f"{base_name}{suffix}.jpg"
        lbl_filename = f"{base_name}{suffix}.txt"

        cv2.imwrite(os.path.join(img_dir, img_filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        with open(os.path.join(lbl_dir, lbl_filename), 'w') as f:
            for cls, box in zip(classes, bboxes):
                f.write(f"{cls} " + " ".join(f"{x:.6f}" for x in box) + "\n")

    # 2. Основной цикл обработки — на каждое изображение сразу сохраняем всё в диск
    total_saved = 0
    for img_path in glob.glob(f"{imgs_folder}/*.jpg"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if img_name not in split_map:
            continue  # на всякий случай, если имя не попало в разбиение

        split = split_map[img_name]
        label_path = os.path.join(annots_folder, f"{img_name}.txt")
        if not os.path.exists(label_path):
            print(f"Пропуск {img_name}: нет аннотации")
            continue

        # Читаем аннотации
        bboxes, classes = [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    classes.append(int(parts[0]))
                    bboxes.append(parts[1:5])
        if not bboxes:
            print(f"Пропуск {img_name}: пустые аннотации")
            continue

        # Загружаем изображение
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Сохраняем оригинал
        save_sample(image, bboxes, classes, split, img_name)
        total_saved += 1

        # Генерируем аугментации и сразу сохраняем
        for i in range(augments):
            try:
                aug = transform(image=image, bboxes=bboxes, class_labels=classes)
                # Фильтруем валидные боксы (на случай выхода за границы)
                valid = [(cls, box) for cls, box in zip(aug['class_labels'], aug['bboxes'])
                        if all(0 <= v <= 1 for v in box)]
                if not valid:
                    print(f"Предупреждение: все bbox невалидны для {img_name}_aug{i}")
                    continue

                valid_classes, valid_bboxes = zip(*valid)
                save_sample(aug['image'], list(valid_bboxes), list(valid_classes), split, img_name, f"_aug{i}")
                total_saved += 1
            except Exception as e:
                print(f"Ошибка при аугментации {img_name}_aug{i}: {e}")

        print(f"Обработан: {img_name} (оригинал + аугментации)")

    print(f"\nВсего сохранено файлов (изображений): {total_saved}")

    # 3. Создаём dataset.yaml
    dataset_config = {
        'path': './augmented',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 3,
        'names': ['apple', 'tomato', 'cucumber']
    }
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)