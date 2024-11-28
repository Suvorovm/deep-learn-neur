from ultralytics import YOLO
import os

# Директория с изображениями
base_dir = r"E:\Maga\2_1\deepLearning\proj\lab4\animals\train"
label_dir = r"E:\Maga\2_1\deepLearning\proj\lab5_n\labels"
os.makedirs(label_dir, exist_ok=True)

# Загружаем предобученную модель YOLOv8
model = YOLO("yolov8s.pt")  # Используем предобученные веса

# Функция для генерации аннотаций
def generate_annotations(image_dir, label_dir, model):
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Обработка JPG и PNG
                img_path = os.path.join(root, file)
                results = model(img_path)

                # Создаем аннотацию в формате YOLO
                label_path = os.path.join(
                    label_dir, os.path.relpath(img_path, image_dir).replace(".jpg", ".txt").replace(".png", ".txt")
                )
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                with open(label_path, "w") as f:
                    for box in results[0].boxes:
                        cls_id = int(box.cls)  # Класс
                        x_center, y_center, width, height = box.xywh[0].tolist()  # Нормализованные координаты
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                print(f"Аннотация создана для {img_path}")



# Генерация аннотаций
generate_annotations(base_dir, label_dir, model)



print(f"Все аннотации сохранены в: {label_dir}")
