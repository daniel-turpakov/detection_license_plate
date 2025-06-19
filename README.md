## Vehicle License Plate Detection and Recognition System
### Project Description
This project aims to develop a comprehensive system that performs two key functions related to vehicle license plates:
* License Plate Detection: The system will identify and locate all license plates present on a cropped image of a vehicle.
* License Plate Recognition (LPR): For each detected license plate, the system will perform optical character recognition (OCR) to read and interpret the characters (both letters and numbers) sequentially.

### How to use:
1. ```git clone https://github.com/daniel-turpakov/detection_license_plate.git```
2. ```cd detection_license_plate```
3. ```pip3 install -r requirements.txt```
4. ```unzip -q weights/best_30ep.zip```
5. ```python3 main.py --image_path <path_to_image> --output <save_path>```

### Example:
```python3 main.py --image_path data/raw_dataset/images/test/8.jpg --output results/result.jpg```

Result: 
![Скриншот](./results/result.jpg)
