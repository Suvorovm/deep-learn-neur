import random
from transformers import BertTokenizer
import torch
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification

# Проверяем доступность GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Available GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU')
    device = torch.device("cpu")

# Загружаем сохраненную модель и токенизатор
model_dir = './model_save_toxic/'
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

model.to(device)

# Загружаем данные
df = pd.read_csv("./data/labeled_rutoxic.csv", delimiter=',', header=0, names=['sentence', 'label'])
sentences = df.sentence.values
labels = df.label.values

# Случайный выбор предложений
count_elements = 30
selected_indices = np.random.choice(len(sentences), count_elements, replace=False)
valid = sentences[selected_indices]
valid_labels = labels[selected_indices]

# Токенизация предложений
input_ids = np.zeros((count_elements, 128), dtype=int)
for i, sentence in enumerate(valid):
    encoded_sentence = tokenizer.encode(
        sentence,
        add_special_tokens=True,  # Добавляем специальные токены [CLS] и [SEP]
        padding='max_length',  # Дополняем до max_length
        max_length=128,  # Максимальная длина
        truncation=True  # Обрезаем предложения длиннее max_length
    )
    input_ids[i] = encoded_sentence

# Создание масок внимания
attention_masks = (input_ids > 0).astype(int)

# Преобразование данных в тензоры
input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).to(device)
attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long).to(device)

# Получение предсказаний модели
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids_tensor, attention_mask=attention_masks_tensor)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

# Вычисление точности
accuracy = (predictions == valid_labels).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Декодирование и вывод результатов
for sentence, prediction, true_label in zip(valid, predictions, valid_labels):
    pred_result = "токсичный" if prediction == 1 else "не токсичный"
    true_result = "токсичный" if true_label == 1 else "не токсичный"
    print(f"Предложение: {sentence}")
    print(f"Предсказание: {pred_result}, Истинная метка: {true_result}\n")
