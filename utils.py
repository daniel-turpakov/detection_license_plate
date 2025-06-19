from pathlib import Path
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def draw_bbox(image_path, bboxs, color=(255, 0, 0), show=False, save_path=None, texts=None):
    """
    Рисует прямоугольник на изображении и отображает/сохраняет результат.
    Поддерживает китайские иероглифы и другие Unicode-символы через PIL.

    :param image_path: путь к изображению
    :param bboxs: координаты [x1, y1, x2, y2]
    :param color: цвет BGR
    :param show: показывать ли изображение
    :param save_path: путь для сохранения (None - не сохранять)
    :param texts: подписи к боксам (поддерживает Unicode)

    :return: None
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение не найдено по пути: {image_path}")

    # Конвертация в RGB (для совместимости с PIL)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)

    font_path = "fonts/simhei.ttf"
    font_size = 30

    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # Рисование прямоугольников и текста
    for i, bbox in enumerate(bboxs):
        x1, y1, x2, y2 = map(int, bbox)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Добавляем текст (через PIL)
        if texts is not None:
            text = texts[i]
            draw.text((x1, y1 - 30), text, font=font, fill=color)  # y1 - 30 чтобы текст был над боксом

    # Конвертируем обратно в массив numpy (обновляем image_rgb)
    image_rgb = np.array(img_pil)

    # Сохранение (если указан путь)
    if save_path:
        plt.imsave(save_path, image_rgb)

    # Отображение (через matplotlib)
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    return None


def generate_yaml_config(dataset_path, path_to_train, path_to_val, path_to_config):
    """
    Генерирует YAML конфиг для датасета, содержащий относительные пути к тренировочной и валидационной выборкам, а также карту имен классов.

    :param dataset_path: путь к основной папке датасета.
    :param path_to_train: абсолютный путь к папке с тренировочными данными.
    :param path_to_val: абсолютный путь к папке с валидационными данными.

    :return absolute_path_to_config: абсолютный путь к созданному YAML конфигурационному файлу.
    """
    config = {
        'path': dataset_path,
        'train': path_to_train,
        'val': path_to_val,
        'names': {
            '0': 'license_plate'
            },

    }

    with open(path_to_config, 'w') as file:
        yaml.dump(config, file)
    absolute_path_to_config = path_to_config.resolve()
    return absolute_path_to_config


def set_params(config_path, scale, anchors, nc):
    """
    Обновляет конфигурационный файл YAML, устанавливая новое значение 'scale'.

    :param config_path: путь к конфигурационному файлу YAML
    :param scale: новое значение размера модели, которое нужно установить

    :return: None
    """
    with open(config_path) as f:
        cur_config = yaml.safe_load(f, )
    cur_config["nc"] = nc
    cur_config["scale"] = scale
    cur_config["anchors"] = anchors

    with open(config_path, 'w') as f:
        yaml.dump(cur_config, f)


def yolo_to_xyxy(txt_path, img_path):
    """
    Преобразует координаты из YOLO-формата (нормализованные x_center, y_center, w, h)
    в абсолютные координаты (x1, y1, x2, y2) для изображения.

    :param txt_path: Путь к файлу разметки YOLO (например, '123.txt').
    :param img_path: Путь к изображению (например, '123.jpg').

    :return: boxes: Список кортежей в формате (class_id, x1, y1, x2, y2).
    """

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Изображение {img_path} не найдено!")
    img_h, img_w = img.shape[:2]

    # Читаем файл разметки YOLO
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        # Пропускаем некорректные строки
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:5])

        # Преобразуем нормализованные координаты в абсолютные
        x_center *= img_w
        y_center *= img_h
        w *= img_w
        h *= img_h

        # Вычисляем координаты углов
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def create_detector_configs(path_to_dataset, path_to_train_images, path_to_val_images, path_to_config):
    """
    Создает конфигурационные файлы для модели YOLOv11l

    :param path_to_dataset:
    :param path_to_train_images:
    :param path_to_val_images:
    :param path_to_config:

    :return: None
    """

    generate_yaml_config(path_to_dataset, path_to_train_images, path_to_val_images, path_to_config)

    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    set_params("configs/yolo11.yaml", "l", anchors, 1)


if __name__ == '__main__':
    path_to_dataset = "data/dataset_detector"
    path_to_train_images = "images/train"
    path_to_val_images = "images/val"
    path_to_config = Path("configs", "data.yaml")

    create_detector_configs(path_to_dataset, path_to_train_images, path_to_val_images, path_to_config)
