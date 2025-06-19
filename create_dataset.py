import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image


def convert_to_yolo(annotations_file, img_dir, output_label_dir):
    """
    Преобразование сырых данных в YOLO-формат

    :param annotations_file: путь к файлу с аннотациями
    :param img_dir: путь к директории с изображениями
    :param output_label_dir: путь к конечной директории

    :return: None
    """
    df = pd.read_csv(annotations_file)

    os.makedirs(output_label_dir, exist_ok=True)

    for img_name, group in df.groupby('image_name'):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img_w, img_h = img.size

        with open(os.path.join(output_label_dir, img_name.split("/")[1].replace('.jpg', '.txt')), 'w') as f:
            for _, row in group.iterrows():

                x_center = ((row['x_1'] + row['x_2']) / 2) / img_w
                y_center = ((row['y_1'] + row['y_2']) / 2) / img_h
                width = (row['x_2'] - row['x_1']) / img_w
                height = (row['y_2'] - row['y_1']) / img_h

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def move_image_and_label(input_dir, image_filenames, is_train):
    """
    Перемещает изображение и аннотации к нему в тренировочную или
    тестовую директорию

    :param input_dir: путь к исходной директории
    :param image_filenames: имена файлов изображений
    :param is_train: являются ли текущий набор данных тренировочным

    :return: None
    """
    subdir = "train" if is_train else "val"

    for train_image_filename in image_filenames:
        try:
            image_name = train_image_filename.split("/")[-1]
            destination_image_filename = f"data/dataset/images/{subdir}/{image_name}"
            shutil.copy2(train_image_filename, destination_image_filename)

            source_label_filename = f"{input_dir}/annotation/all_labels/{image_name.replace('.jpg', '.txt')}"
            destination_label_filename = f"data/dataset/labels/{subdir}/{image_name.replace('.jpg', '.txt')}"
            shutil.copy2(source_label_filename, destination_label_filename)
        except:
            continue


def create_dataset(input_dir, train_part_size=0.8):
    """
    Создает тренировочные и тестовые данные в YOLO-формате

    :param input_dir: путь к исходной директории
    :param train_part_size: размер доли тренировочных данных

    :return: None
    """

    np.random.seed(42)

    os.makedirs("data/dataset/images/train", exist_ok=True)
    os.makedirs("data/dataset/images/val", exist_ok=True)
    os.makedirs("data/dataset/labels/train", exist_ok=True)
    os.makedirs("data/dataset/labels/val", exist_ok=True)

    all_images_filenames = sorted([f"{input_dir}/images/train/{filename}"
                                   for filename in os.listdir(f"{input_dir}/images/train")])
    total_images_num = len(all_images_filenames)

    shuffled_indices = np.random.permutation(total_images_num)
    split_idx = int(train_part_size * total_images_num)
    train_images_filenames = [all_images_filenames[i] for i in shuffled_indices[:split_idx]]
    test_images_filenames = [all_images_filenames[i] for i in shuffled_indices[split_idx:]]

    move_image_and_label(input_dir, train_images_filenames, is_train=True)
    move_image_and_label(input_dir, test_images_filenames, is_train=False)


if __name__ == '__main__':
    convert_to_yolo("data/raw_dataset/annotation/train_final_annot.txt",
                    'data/raw_dataset/images',
                    'data/raw_dataset/annotation/all_labels')

    create_dataset("data/raw_dataset")
