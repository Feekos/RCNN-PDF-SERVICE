import os
import fitz
import torch
from PIL import Image
import cv2
import numpy as np
import json
from unet_model import UNet
from torchvision import transforms
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3ForTokenClassification

# === Пути ===
PDF_PATH = "sbornik_1.pdf"
IMAGES_DIR = "enhanced_images"
ANNOTATIONS_DIR = "annotations"
MODEL_PATH = "unet_publaynet.pth"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# === Настройки модели ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {
    0: 'background',
    1: 'text',
    2: 'title',
    3: 'author',
    4: 'figure',
    5: 'table',
    6: 'bibliography'
}

# === Трансформации ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === Загрузка U-Net ===
def load_unet_model(model_path, n_classes=7):
    model = UNet(n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# === Сегментация страницы ===
def segment_page_with_unet(image_path, model):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    print(f"✅ Маска получена для {image_path}")
    return image, mask

# === Генерация аннотаций на основе маски ===
def generate_annotation_from_mask(mask, image_size=(512, 512)):
    annotations = []

    for cls_id, label in CLASSES.items():
        if cls_id == 0:
            continue  # Пропуск фона

        coords = cv2.findNonZero((mask == cls_id).astype(np.uint8))
        if coords is None or len(coords) == 0:
            continue

        x, y, w, h = cv2.boundingRect(coords)
        annotations.append({
            "original_width": image_size[0],
            "original_height": image_size[1],
            "image_rotation": 0,
            "value": {
                "x": int(x / image_size[0] * 100),
                "y": int(y / image_size[1] * 100),
                "width": int(w / image_size[0] * 100),
                "height": int(h / image_size[1] * 100),
                "rotation": 0
            },
            "from_name": "label",
            "to_name": "image",
            "type": "RectangleLabels",
            "value": {"rectanglelabels": [label]}
        })

    return annotations

# === Обнаружение layout через LayoutLMv3 (опционально) ===
def detect_layout_elements(image_path):
    feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(DEVICE)

    image = Image.open(image_path).convert("RGB")
    encoding = feature_extractor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = feature_extractor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    boxes = encoding["bbox"][0]
    labels = predictions[0]

    elements = []
    for token, box, label_id in zip(tokens, boxes, labels):
        if token.startswith("##") or token in ["[CLS]", "[SEP]"]:
            continue
        label = model.config.id2label[label_id.item()]
        elements.append({
            "token": token,
            "box": box.tolist(),
            "label": label
        })

    return elements

# === Сохранение аннотации в формате Label Studio ===
def save_label_studio_annotation(image_name, annotation_data):
    result = [{
        "id": "result1",
        "from_name": "label",
        "to_name": "image",
        "type": "RectangleLabels",
        "value": data
    } for data in annotation_data]

    annotation = {
        "data": {"image": f"/data/local-files/?d={image_name}"},
        "predictions": [{
            "model_version": "layout-unet-v1",
            "result": result
        }]
    }

    filename = os.path.splitext(os.path.basename(image_name))[0] + ".json"
    with open(os.path.join(ANNOTATIONS_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)

# === Основной процесс генерации аннотаций ===
if __name__ == "__main__":
    print("🚀 Начало генерации аннотаций...")

    # Шаг 2: Загрузка модели
    unet_model = load_unet_model(MODEL_PATH)

    # Шаг 3: Обработка всех страниц
    for filename in os.listdir(IMAGES_DIR):
        if filename.endswith(".png"):
            image_path = os.path.join(IMAGES_DIR, filename)
            image, mask = segment_page_with_unet(image_path, unet_model)

            # Шаг 4: Генерация аннотаций
            annotation_data = generate_annotation_from_mask(mask)

            # Шаг 5: Сохранение в формате Label Studio
            save_label_studio_annotation(filename, annotation_data)

    print(f"✅ Аннотации сохранены в папке {ANNOTATIONS_DIR}")