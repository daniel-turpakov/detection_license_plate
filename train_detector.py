from ultralytics import YOLO


def train_detector(model_config, pretrained_weights, data_config, train_config):
    """
    Запускает тренировку модели YOLO с заданными параметрами

    :param model_config: путь к конфигурационному файлу модели
    :param pretrained_weights: путь к предобученным весам модели
    :param data_config: путь к конфигурационному файлу данных для обучения
    :param train_config: параметры для тренировки

    :return: None
    """
    model = YOLO(model_config).to('cuda')
    model.load(pretrained_weights)

    model.train(
        data=data_config,
        **train_config
    )


if __name__ == '__main__':
    model_conf = "configs/yolo11.yaml"
    raw_weights = "weights/yolo11l.pt"
    data_conf = "configs/data.yaml"

    train_conf = {
        "epochs": 30,
        "imgsz": 640,
        "batch": 12,
        "patience": 20,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 3e-5,
        "augment": False
    }

    train_detector(model_conf, raw_weights, data_conf, train_conf)
