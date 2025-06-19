from ultralytics import YOLO
from tqdm import tqdm
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def inference_detector(yolo_model, test_images_dir):
    """
    Запускает детектор и создает файл с аннотациями для тестовых изображений

    :param yolo_model: обученная модель YOLO
    :param test_images_dir: путь к директории с тестовыми изображениями

    :return: None
    """

    test_images = [f"{test_images_dir}/{test_image}" for test_image in sorted(os.listdir(test_images_dir))]
    with open("test_pred_bboxes.txt", "w") as wf:
        wf.write("image_name,x_1,y_1,x_2,y_2,conf" + "\n")
        for test_image_name in tqdm(test_images):
            image = cv2.imread(test_image_name)
            image_name = f"test/{test_image_name.split('/')[-1]}"
            result = yolo_model.predict(source=image, save=False, conf=0.3)[0]
            bboxes = result.boxes.xyxy
            confs = result.boxes.conf
            if len(bboxes) == 0:
                    continue
            for i, bbox in enumerate(bboxes):
                bbox = bbox.cpu().numpy()
                x1, y1, x2, y2 = bbox
                wf.write(f"{image_name},{x1},{y1},{x2},{y2},{confs[i]}" + "\n")


def predict_and_show_image(yolo_model, image_path):
    """
    Запускает детектор и выводит изображение с предсказанным результатом

    :param yolo_model: обученная модель YOLO
    :param image_path: путь к изображению

    :return: None
    """

    image = cv2.imread(image_path)
    results = yolo_model.predict(source=image, save=True)
    res = cv2.imread(Path(results[0].save_dir, results[0].path))
    plt.imshow(res[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    trained_model = YOLO("weights/best_30ep.pt")
    inference_detector(trained_model, "data/raw_dataset/images/test")
