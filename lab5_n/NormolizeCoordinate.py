import os


def normalize_labels_in_folder(label_folder, image_width, image_height):
    """
    Нормализует координаты всех аннотаций в папке label_folder.
    label_folder: Путь к папке с текстовыми файлами аннотаций.
    image_width: Ширина изображений.
    image_height: Высота изображений.
    """
    for label_file in os.listdir(label_folder):
        label_path = os.path.join(label_folder, label_file)

        # Проверяем, что это текстовый файл
        if not label_file.endswith(".txt"):
            continue

        normalized_lines = []
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            values = list(map(float, line.strip().split()))
            if len(values) != 5:
                print(f"Некорректная строка в {label_file}: {line}. Пропускаем.")
                continue

            cls, x, y, w, h = values
            x /= image_width
            y /= image_height
            w /= image_width
            h /= image_height

            # Убедимся, что значения находятся в пределах [0, 1]
            if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                normalized_lines.append(f"{cls} {x} {y} {w} {h}\n")
            else:
                print(f"Координаты объекта в {label_file} выходят за пределы [0, 1]: {line}")

        # Перезаписываем файл с нормализованными данными
        with open(label_path, 'w') as f:
            f.writelines(normalized_lines)

        print(f"Файл {label_file} успешно нормализован.")


# Укажите путь к папке с текстовыми файлами и размеры изображений
label_folder = r"E:\Maga\2_1\deepLearning\proj\lab5_n\data\val\labels"
image_width = 520  # Укажите ширину изображений
image_height = 520  # Укажите высоту изображений

normalize_labels_in_folder(label_folder, image_width, image_height)
