"""Бенчмарк CurrentYoloVitModel на датасете Micro_Facial_Expressions.

Запуск из корня проекта:  python benchmark.py
"""

import sys
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# Корень проекта = директория этого файла (benchmark.py лежит в корне).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mood_base.models.current_yolo_vit import CurrentYoloVitModel
from contracts.schemas import Emotion

DATASET_NAME = "LaurenGurgiolo/Micro_Facial_Expressions"
# Путь к модели относительно корня проекта (тот же, что в pipeline_registry).
MODEL_PATH = "mood_base/model/yolov8n-face.pt"
PER_CLASS = 150  # Сколько примеров максимум брать на КАЖДЫЙ класс (ровный баланс)

# Датасет — смесь двух источников (FER-2013 + micro-expression) с разным
# именованием, поэтому метки содержат синонимы (anger/happy-дубли и т.п.).
# Приводим их к каноничному словарю ViT-модели / Emotion.
LABEL_ALIASES = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
}

def collect_stratified(ds, label_names, per_class):
    """Собирает до `per_class` примеров на каждый класс из стрима.

    Возвращает список (item, canonical_label). Так как классы в стриме идут
    блоками, проходим весь сплит и режем по каждому классу отдельно, затем
    перемешиваем — иначе выборка была бы несбалансированной.
    """
    buckets = defaultdict(list)
    for item in tqdm(ds, desc="Стратифицированный отбор"):
        raw = label_names[int(item["label"])]
        canonical = LABEL_ALIASES.get(raw, raw)
        if len(buckets[canonical]) < per_class:
            buckets[canonical].append((item, canonical))

    # Перемешиваем внутри каждого класса и объединяем
    samples = []
    for canonical, items in buckets.items():
        random.shuffle(items)
        samples.extend(items)

    random.shuffle(samples)
    return samples


def main():
    print(f"Загрузка датасета {DATASET_NAME} (сплит: test)...")
    ds = load_dataset(DATASET_NAME, split="test", streaming=True)

    # Маппинг индекс -> имя эмоции (0->angry, 1->disgust, 2->fear, ...).
    # Датасет отдаёт label как ClassLabel (int), а не строку.
    label_names = [n.lower() for n in ds.features["label"].names]

    print(f"Стратифицированный отбор: до {PER_CLASS} примеров на класс...")
    dataset = collect_stratified(ds, label_names, PER_CLASS)

    # Отчёт о составе выборки
    counts = defaultdict(int)
    for _, label in dataset:
        counts[label] += 1
    print("\nСостав сбалансированной выборки:")
    for label in sorted(counts):
        print(f"  {label:<10}: {counts[label]}")
    print(f"  {'ВСЕГО':<10}: {len(dataset)}\n")
    
    print("Инициализация CurrentYoloVitModel...")
    # Обязательно analyze_every_n_frames=1, так как кадры не связаны
    model = CurrentYoloVitModel(
        model_path=MODEL_PATH,
        analyze_every_n_frames=1
    )
    
    y_true = []
    y_pred = []

    print("Инференс через пайплайн...")
    for frame_id, (item, true_label) in enumerate(tqdm(dataset, total=len(dataset))):
        # HF отдает PIL Image. Конвертируем в BGR numpy array для YOLO
        pil_img = item["image"].convert("RGB")
        frame_bgr = np.array(pil_img)[:, :, ::-1].copy()
        
        y_true.append(true_label)
        
        # Вызываем контракт
        result = model.predict(frame_id=frame_id, frame=frame_bgr)
        
        if not result.detections:
            # YOLO не нашел лицо на кадре
            y_pred.append(Emotion.UNKNOWN.value)
        else:
            # Берем самое уверенное лицо в кадре
            best_det = max(result.detections, key=lambda d: d.face_confidence)
            y_pred.append(best_det.emotion.value)

    # --- Подсчет метрик ---
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n--- Результаты Бенчмарка (CurrentYoloVitModel | Test Split) ---")
    print(f"Macro-F1:     {macro_f1:.4f}")
    print(f"Weighted-F1:  {weighted_f1:.4f}")

    # Per-class метрики — при дисбалансе это ключ к пониманию
    labels = sorted(list(set(y_true) | set(y_pred)))
    print("\n--- Отчёт по классам (precision / recall / f1 / support) ---")
    print(classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    ))

    # --- Отрисовка Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix\nCurrentYoloVitModel on {DATASET_NAME.split("/")[-1]} (test)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_yolo_vit.png', dpi=300)
    print("\nМатрица ошибок сохранена как 'confusion_matrix_yolo_vit.png'")

if __name__ == "__main__":
    main()