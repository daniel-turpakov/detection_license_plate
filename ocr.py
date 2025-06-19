import cv2
from paddleocr import TextRecognition
from tqdm import tqdm


def crop_image_by_box(image_path, box):
    """
    Обрезает изображение по заданному bounding box

    :param image_path: путь к изображению
    :param box: координаты bounding box в формате (x1, y1, x2, y2)

    :return cropped_image: обрезанное изображение
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение по пути {image_path} не найдено!")
    x1, y1, x2, y2 = box

    h, w = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"Координаты бокса {box} выходят за границы изображения {w}x{h}!")

    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

    return cropped_image


def get_plate(image_path, box, ocr_model):
    """
    Распознает государственный регистрационный знак
    в зоне бокса, который предсказал детектор

    :param image_path: путь к исходному изображению
    :param box: предсказанный детектором бокс
    :param ocr_model: модель для распознавания символов

    :return plate: государственный регистрационный знак
    """
    cropped_plate = crop_image_by_box(image_path, box)
    output = ocr_model.predict(input=cropped_plate, batch_size=1)
    if not output:
        return None
    res = output[0]
    plate = res.get("rec_text")
    plate.replace(" ", "")
    return plate


def inference_ocr(test_images_dir, pred_boxes_file, result_plate_file, paddle_model_name):
    """
    Запускает ocr и создает файл с аннотациями для тестовых изображений

    :param test_images_dir: путь к директории с тестовыми изображениями
    :param pred_boxes_file: путь к файлу с результатами детектора
    :param result_plate_file: путь к файлу с результатами работы ocr
    :param paddle_model_name: название модели PaddleOCR

    :return: None
    """
    paddle_model = TextRecognition(model_name=paddle_model_name)
    with open(pred_boxes_file) as rf:
        all_bboxes = rf.readlines()[1:]
    with open(result_plate_file, "w") as wf:
        wf.write("image_name,plate\n")
        for row in tqdm(all_bboxes):
            image_rel_path, x1, y1, x2, y2, _ = row.split(",")
            box = (float(x1), float(y1), float(x2), float(y2))
            full_image_path = f"{test_images_dir}/{image_rel_path}"
            predicted_plate = get_plate(full_image_path, box, paddle_model)
            if predicted_plate:
                wf.write(f"{image_rel_path},{predicted_plate}\n")


if __name__ == "__main__":
    test_images_dir = "data/raw_dataset/images"
    pred_boxes_file = "test_pred_bboxes.txt"
    result_plate_file = "test_pred_plates.txt"
    paddle_model_name = "PP-OCRv5_server_rec"

    inference_ocr(test_images_dir, pred_boxes_file, result_plate_file, paddle_model_name)
