from ultralytics import YOLO
import cv2
from paddleocr import TextRecognition
from ocr import get_plate
from utils import draw_bbox
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')


def get_license_plate(image_path, save_path):
    """
    Детектирует и распознает номера автомобилей


    :param image_path: путь к изображению
    :param save_path: путь, куда сохранится итоговое изображение

    :return all_plates: список всех распознанных номеров
    """

    yolo_model = YOLO("weights/best_30ep.pt")
    paddle_model = TextRecognition(model_name="PP-OCRv5_server_rec")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение по пути {image_path}", file=sys.stderr)
        return None

    detection_result = yolo_model.predict(source=image, save=False, verbose=False)[0]
    bboxes = detection_result.boxes.xyxy
    if len(bboxes) == 0:
        print("Номерные знаки не обнаружены")
        return None

    all_plates = []
    for bbox in bboxes:
        bbox = bbox.cpu().numpy()
        license_plate = get_plate(image_path, bbox, paddle_model)
        all_plates.append(license_plate)

    draw_bbox(image_path, bboxes, texts=all_plates, save_path=save_path, show=False)

    return all_plates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Распознавание автомобильных номеров")
    parser.add_argument("--image_path", help="Путь к изображению для обработки")
    parser.add_argument("--output", help="Путь для сохранения результата")

    args = parser.parse_args()

    # Обработка изображения
    plates = get_license_plate(args.image_path, args.output)

    # Вывод результатов в консоль
    if plates:
        print("Распознанные номера:")
        for i, plate in enumerate(plates, 1):
            print(f"{i}. {plate}")
    else:
        print("Номера не распознаны")
