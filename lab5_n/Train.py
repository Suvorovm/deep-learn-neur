from multiprocessing import freeze_support

from ultralytics import YOLO
def fx():
# Загружаем предобученную модель
    model = YOLO("yolov8s.pt")

    # Fine-tuning модели с сохранением в текущей директории
    model.train(
        data="E:/Maga/2_1/deepLearning/proj/lab5_n/masked.yaml",
        epochs=3,
        imgsz=520,
        batch=16,
        project=".",
        name="trained_model"  # Имя папки для модели
    )

if __name__ == '__main__':
    freeze_support()
    fx()