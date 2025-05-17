import os
import torch
from PIL import Image
import cv2
import numpy as np
import json
import time
import logging
import matplotlib.pyplot as plt
from unet_model import UNet
from torchvision import transforms
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from collections import defaultdict
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

# === Настройка логгирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Пути ===
IMAGES_ROOT_DIR = "enhanced_images"  # Корневая папка с подкаталогами
ANNOTATIONS_DIR = "annotations"
MODEL_PATH = "unet_publaynet.pth"

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# === КЛАССЫ ДЛЯ СЕГМЕНТАЦИИ ===
CLASSES = {
    0: 'background',
    1: 'text',
    2: 'title',
    3: 'author',
    4: 'figure',
    5: 'table',
    6: 'bibliography'
}

LABEL_MAP_LAYOUTLM_TO_CLASSES = {
    "Text": "text",
    "Title": "title",
    "Author": "author",
    "Figure": "figure",
    "Table": "table",
    "Bibliography": "bibliography",
    "PageFooter": "figure",  # Маппинг редких классов
    "PageHeader": "header",  # Можно добавить отдельно или объединить
    "Footnote": "bibliography",
    "Formula": "text",
    "List-item": "text",
    "Caption": "title",
    "Section-header": "title",
    "Abstract": "text",
    "Document-header": "title",
    "Document-footer": "footer",
    "Picture": "figure",
    "Other": "background"
}

# === ТРАНСФОРМАЦИИ ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === УСТРОЙСТВО ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === ЗАГРУЗКА МОДЕЛИ U-NET ===
def load_unet_model(model_path, n_classes=len(CLASSES)):
    logger.info("🧠 Загрузка модели U-Net...")
    model = UNet(n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# === СЕГМЕНТАЦИЯ С ПОМОЩЬЮ U-NET ===
def segment_page_with_unet(image_path, model):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    logger.debug(f"✅ Маска получена для {image_path}")
    return image, mask


# === ГЕНЕРАЦИЯ АННОТАЦИЙ ПО МАСКЕ ===
def generate_annotation_from_mask(mask, image_size=(512, 512), min_box_size=5):
    """
    Генерирует аннотации по маске с учётом минимального размера объекта
    """
    annotations = []
    for cls_id, label in CLASSES.items():
        if cls_id == 0:
            continue  # Пропуск фона

        coords = cv2.findNonZero((mask == cls_id).astype(np.uint8))
        if coords is None or len(coords) < 2:
            continue

        x, y, w, h = cv2.boundingRect(coords)

        # Фильтрация слишком маленьких регионов
        if w < min_box_size or h < min_box_size:
            continue

        annotation = {
            "id": f"result_{cls_id}",
            "from_name": "label",
            "to_name": "image",
            "type": "RectangleLabels",
            "value": {
                "x": int(x / image_size[0] * 100),
                "y": int(y / image_size[1] * 100),
                "width": int(w / image_size[0] * 100),
                "height": int(h / image_size[1] * 100),
                "rotation": 0,
                "rectanglelabels": [label]
            }
        }

        annotations.append(annotation)

    return annotations


# === ИСПОЛЬЗОВАНИЕ LayoutLMv3 (резервный вариант) ===
def detect_layout_elements(image_path):
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    if "word_boxes" in encoding:
        del encoding["word_boxes"]

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    boxes = encoding["bbox"][0].cpu().numpy()
    labels = predictions[0].cpu().numpy()

    width, height = image.size
    elements = []

    for token, box, label_id in zip(tokens, boxes, labels):
        raw_label = model.config.id2label[label_id]

        # Маппинг в ваши классы
        mapped_label = LABEL_MAP_LAYOUTLM_TO_CLASSES.get(raw_label, None)

        # Вывод диагностики
        logger.debug(f"🔍 Токен: {token}, Raw Label: {raw_label}, Mapped Label: {mapped_label}")

        if mapped_label is None:
            continue

        x1, y1, x2, y2 = box
        x1 = int(x1 / 1000 * width)
        y1 = int(y1 / 1000 * height)
        x2 = int(x2 / 1000 * width)
        y2 = int(y2 / 1000 * height)
        w = x2 - x1
        h = y2 - y1

        # Убираем очень маленькие регионы
        if w < 2 or h < 2:
            continue

        elements.append({
            "token": token,
            "box": [x1, y1, x2, y2],
            "raw_label": raw_label,
            "mapped_label": mapped_label
        })

    logger.info(f"📄 Найдено элементов на странице {image_path}: {len(elements)}")
    return elements


# === СОХРАНЕНИЕ АННОТАЦИИ Label Studio ===
def save_label_studio_annotation(image_name, image_path, annotation_data):
    result = [{
        "id": f"result_{i + 1}",
        "from_name": "label",
        "to_name": "image",
        "type": "RectangleLabels",
        "value": data["value"]
    } for i, data in enumerate(annotation_data)]

    # Получаем имя подкаталога и номер страницы
    relative_path = os.path.relpath(image_path, start=IMAGES_ROOT_DIR)
    parts = relative_path.split(os.sep)
    if len(parts) < 2:
        raise ValueError(f"❌ Неверная структура пути: {image_path}")

    subfolder = parts[0]
    filename = os.path.splitext(os.path.basename(image_name))[0]  # 'enhanced_page_0'

    final_filename = f"{subfolder}_{filename}.json"
    path = os.path.join(ANNOTATIONS_DIR, final_filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "data": {"image": f"/data/local-files/?d={image_name}"},
            "predictions": [{
                "model_version": "layout-unet-v1",
                "result": result
            }]
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Аннотация сохранена: {final_filename} | Найдено элементов: {len(result)}")
    return len(result)


# === РЕКУРСИВНЫЙ ПОИСК ИЗОБРАЖЕНИЙ ===
def find_all_images(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".png"):
                image_paths.append(os.path.join(root, file))
    return image_paths


# === ОСНОВНОЙ ЦИКЛ ===
if __name__ == "__main__":
    start_time = time.time()
    logger.info("🚀 Начало генерации аннотаций...")

    # Шаг 1: Проверяем наличие изображений
    all_image_paths = find_all_images(IMAGES_ROOT_DIR)
    if not all_image_paths:
        logger.error("🖼️ Ни одного изображения не найдено. Проверьте структуру каталогов.")
        exit()

    # Шаг 2: Загрузка модели
    unet_model = None
    if os.path.exists(MODEL_PATH):
        unet_model = load_unet_model(MODEL_PATH)

        # Диагностика: проверяем работу модели на тестовом изображении
        test_image_path = all_image_paths[0]
        try:
            image, mask = segment_page_with_unet(test_image_path, unet_model)
            print("Test mask shape:", mask.shape)
            print("Unique values in test mask:", np.unique(mask))
        except Exception as e:
            logger.warning(f"⚠️ Предупреждение: ошибка валидации модели U-Net: {e}")
    else:
        logger.warning("⚠️ Модель U-Net не найдена. Аннотации будут сгенерированы с помощью LayoutLMv3.")

    # Шаг 3: Обработка всех страниц во всех подкаталогах
    total_pages = 0
    total_annotations = 0
    class_distribution = defaultdict(int)

    for img_path in all_image_paths:
        try:
            logger.info(f"📄 Обработка: {img_path}")

            if unet_model is not None:
                image, mask = segment_page_with_unet(img_path, unet_model)
                annotation_data = generate_annotation_from_mask(mask)
            else:
                layout_elements = detect_layout_elements(img_path)
                annotation_data = [
                    {
                        "value": {
                            "x": int(elem['box'][0] / 512 * 100),
                            "y": int(elem['box'][1] / 512 * 100),
                            "width": int((elem['box'][2] - elem['box'][0]) / 512 * 100),
                            "height": int((elem['box'][3] - elem['box'][1]) / 512 * 100),
                            "rotation": 0,
                            "rectanglelabels": [elem['mapped_label']]
                        },
                        "from_name": "label",
                        "to_name": "image",
                        "type": "RectangleLabels"
                    }
                    for elem in layout_elements
                ]

            num_annots = save_label_studio_annotation(os.path.basename(img_path), img_path, annotation_data)
            total_annotations += num_annots
            total_pages += 1

            # Подсчёт распределения классов
            for ann in annotation_data:
                label = ann["value"]["rectanglelabels"][0]
                class_distribution[label] += 1

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке файла {img_path}: {e}")


    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)

    logger.info("📊 Метрики:")
    logger.info(f"📄 Всего обработано страниц: {total_pages}")
    logger.info(f"📌 Всего найдено элементов: {total_annotations}")
    logger.info(f"⏱️ Общее время выполнения: {elapsed_time} секунд")
    logger.info("🧩 Распределение по классам:")
    for label, count in class_distribution.items():
        logger.info(f"   - {label}: {count}")