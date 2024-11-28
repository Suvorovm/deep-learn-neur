from ultralytics import YOLO
import os
import cv2

# Путь к папке с изображениями
base_dir = r"E:\Maga\2_1\deepLearning\proj\lab4\animals\val"
# Папка для сохранения обработанных изображений
output_dir = r"E:\Maga\2_1\deepLearning\proj\lab5_n\output"

# Убедимся, что папка для сохранения существует
os.makedirs(output_dir, exist_ok=True)
#trained_model5 - норм
# Загрузка модели YOLO с предобученными весами
model = YOLO(r"E:\Maga\2_1\deepLearning\proj\lab5_n\trained_model\weights\best.pt")

def generate_info(image_dir, model, dir_to_save):
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Обработка JPG и PNG
                img_path = os.path.join(root, file)

                # Получение результатов модели
                results = model(img_path)

                # Загрузка изображения
                img = cv2.imread(img_path)

                # Инициализация переменных для хранения объекта с наибольшей вероятностью
                max_confidence = 0
                best_box = None
                best_class = None

                # Поиск объекта с максимальной вероятностью
                for result in results:
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        if conf > max_confidence:
                            max_confidence = conf
                            best_box = box
                            best_class = int(box.cls[0].item())

                # Если найден объект с максимальной вероятностью, наносим его аннотацию
                if best_box is not None:
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    class_name = model.names[best_class]

                    # Рисование прямоугольника
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Вычисление центра прямоугольника
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Добавление метки по центру квадрата
                    label = f"{class_name} {max_confidence:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y + text_size[1] // 2
                    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Сохранение обработанного изображения
                output_path = os.path.join(dir_to_save, f"annotated_{file}")
                cv2.imwrite(output_path, img)
                print(f"Обработанное изображение сохранено: {output_path}")

# Запускаем обработку
generate_info(base_dir, model, output_dir)
