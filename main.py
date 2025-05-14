import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import torch
import torch.nn as nn
from unet_model import UNet
from torchvision import transforms
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3ForTokenClassification
import pytesseract
import networkx as nx
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import re
import json

# === ПУТИ И ФАЙЛЫ ===
PDF_PATH = "sbornik_1.pdf"
OUTPUT_IMAGES_DIR = "images"
ENHANCED_IMAGES_DIR = "enhanced_images"
SEGMENTED_MASKS_DIR = "segmented_masks"
JSON_OUTPUT = "structured_output.json"

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(ENHANCED_IMAGES_DIR, exist_ok=True)
os.makedirs(SEGMENTED_MASKS_DIR, exist_ok=True)

# === КЛАССЫ ДЛЯ СЕГМЕНТАЦИИ ===
CLASSES = {
    0: 'background',
    1: 'text',
    2: 'title',
    3: 'author',
    4: 'table',
    5: 'figure',
    6: 'bibliography'
}

# === ЭТАП 1: КОНВЕРТАЦИЯ PDF В ИЗОБРАЖЕНИЯ ===
def pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_folder, f"page_{page_num}.png")
        pix.save(image_path)
    doc.close()
    print("✅ PDF конвертирован в изображения.")

# === ЭТАП 2: УЛУЧШЕНИЕ КАЧЕСТВА ИЗОБРАЖЕНИЙ ===
def enhance_image_quality(image_path, enhanced_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, None, h=10)
    cv2.imwrite(enhanced_path, denoised)
    print(f"✅ Изображение {image_path} улучшено.")

# === ЭТАП 3: СЕГМЕНТАЦИЯ С U-NET ===
def load_unet_model(model_path="unet_publaynet.pth", n_classes=7):
    model = UNet(n_classes=n_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def segment_page_with_unet(image_path, model, output_mask_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB').resize((512, 512))
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().numpy()

    cv2.imwrite(output_mask_path, (mask * 255 / mask.max()).astype(np.uint8))
    print(f"✅ Маска сохранена: {output_mask_path}")
    return mask

# === ЭТАП 4: АНАЛИЗ LAYOUT С ИСПОЛЬЗОВАНИЕМ LayoutLMv3 ===
def detect_layout_elements(image_path):
    feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    image = Image.open(image_path).convert("RGB")
    encoding = feature_extractor(image, return_tensors="pt")

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

    print("✅ Layout проанализирован с помощью LayoutLMv3.")
    return elements

# === ЭТАП 5: ГРАФОВАЯ МОДЕЛЬ ===
def build_graph(elements):
    G = nx.Graph()
    for i, elem in enumerate(elements):
        G.add_node(i, label=elem['label'], bbox=elem['box'])

    for i in range(len(elements)):
        for j in range(i+1, len(elements)):
            if abs(elements[i]['box'][3] - elements[j]['box'][1]) < 50:
                G.add_edge(i, j)

    print("✅ Граф структуры документа построен.")
    return G

# === ЭТАП 6: OCR ПО РЕГИОНАМ ===
def ocr_by_class(image_array, mask_array):
    image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
    result = {}

    class_ids = {
        2: "title",
        3: "author",
        6: "references"
    }

    for cls_id, name in class_ids.items():
        coords = cv2.findNonZero((mask_array == cls_id).astype(np.uint8))
        if coords is None or len(coords) == 0:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        roi = image[y:y+h, x:x+w]
        result[name] = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))).strip()

    return result

# === ЭТАП 7: КОНТЕКСТНЫЙ АНАЛИЗ ЧЕРЕЗ BERT ===
def bert_context_analysis(texts):
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# === ЭТАП 8: КЛАССИФИКАЦИЯ ЭЛЕМЕНТОВ ===
def classify_elements(embeddings):
    X_train = [[100, 50], [200, 150], [50, 80]]
    y_train = ["header", "table", "image"]

    mlp = MLPClassifier(hidden_layer_sizes=(64,))
    mlp.fit(X_train, y_train)
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)

    predicted_type_mlp = mlp.predict([[120]][[60]])[0]
    predicted_type_lgbm = lgbm.predict([[180]][[140]])[0]

    return predicted_type_mlp, predicted_type_lgbm

# === ЭТАП 9: ЭКСПОРТ В СТРУКТУРИРОВАННЫЕ ФОРМАТЫ ===
def export_data(all_results):
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"✅ Все результаты сохранены в {JSON_OUTPUT}")

# === ЗАПУСК ===
if __name__ == "__main__":
    print("🚀 Начало обработки PDF...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = load_unet_model().to(device)

    # Шаг 1: Конвертация PDF в изображения
    pdf_to_images(PDF_PATH, OUTPUT_IMAGES_DIR)

    # Шаг 2: Улучшение качества изображений
    for filename in os.listdir(OUTPUT_IMAGES_DIR):
        if filename.endswith(".png"):
            input_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
            enhanced_path = os.path.join(ENHANCED_IMAGES_DIR, "enhanced_" + filename)
            enhance_image_quality(input_path, enhanced_path)

    all_results = []

    # Шаг 3: Обработка всех страниц
    for filename in os.listdir(ENHANCED_IMAGES_DIR):
        if filename.startswith("enhanced_page_") and filename.endswith(".png"):
            page_num = int(re.search(r'_(\d+)\.png', filename).group(1))
            enhanced_image_path = os.path.join(ENHANCED_IMAGES_DIR, filename)
            mask_path = os.path.join(SEGMENTED_MASKS_DIR, f"mask_page_{page_num}.png")

            # Шаг 4: Сегментация с использованием U-Net
            mask = segment_page_with_unet(enhanced_image_path, unet_model, mask_path)
            image = Image.open(enhanced_image_path)

            # Шаг 5: OCR по регионам
            segmented_data = ocr_by_class(image, mask)

            # Шаг 6: Анализ layout
            try:
                layout_elements = detect_layout_elements(enhanced_image_path)
            except Exception as e:
                layout_elements = []

            # Шаг 7: Построение графовой модели
            try:
                graph = build_graph(layout_elements)
                graph_data = nx.node_link_data(graph)
            except:
                graph_data = {"nodes": [], "links": []}

            # Шаг 8: Сохраняем данные этой страницы
            title = segmented_data.get("title", "")
            authors = segmented_data.get("author", "").splitlines()
            references = segmented_data.get("references", "")

            all_results.append({
                "page": page_num,
                "metadata": {
                    "title": title,
                    "authors": authors,
                    "references": references
                },
                "layout_elements": layout_elements,
                "graph": graph_data
            })

    # Шаг 9: Сохранение итогового JSON
    export_data(all_results)

    print("🎉 Обработка завершена!")