import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
from unet_model import UNet
from torchvision import transforms
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import pytesseract
import networkx as nx
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import re
import json

# === ПУТИ И ФАЙЛЫ ===
PDF_FILES = ["sbornik_1.pdf", "sbornik_2.pdf", "sbornik_3.pdf"]
OUTPUT_IMAGES_ROOT = "images"
ENHANCED_IMAGES_ROOT = "enhanced_images"
SEGMENTED_MASKS_DIR = "segmented_masks"
JSON_OUTPUT = "structured_output.json"

os.makedirs(OUTPUT_IMAGES_ROOT, exist_ok=True)
os.makedirs(ENHANCED_IMAGES_ROOT, exist_ok=True)
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
    print("✅ PDF конвертирован в изображения PNG.")

# === ЭТАП 2: УЛУЧШЕНИЕ КАЧЕСТВА ИЗОБРАЖЕНИЙ ===
def enhance_single_image(args):
    input_path, enhanced_path = args
    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary, None, h=10)
        cv2.imwrite(enhanced_path, denoised)
        print(f"✅ Изображение {input_path} улучшено.")
    except Exception as e:
        print(f"❌ Ошибка при обработке {input_path}: {e}")

def parallel_enhance_images(input_dir, output_dir, max_workers=4):
    tasks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            enhanced_path = os.path.join(output_dir, "enhanced_" + filename)
            tasks.append((input_path, enhanced_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(enhance_single_image, tasks)

    print("✅ Все изображения улучшены.")

# === ЭТАП 3: СЕГМЕНТАЦИЯ С U-NET ===
def load_unet_model(model_path="unet_publaynet.pth", n_classes=len(CLASSES)):
    if not os.path.exists(model_path):
        print("⚠️ Модель не найдена. Запуск обучения U-Net...")
        try:
            import subprocess
            result = subprocess.run(
                ["python", "train_unet.py"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("✅ Обучение успешно завершено.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("❌ Ошибка при обучении модели:")
            print(e.stderr)
            raise RuntimeError("Не удалось обучить модель U-Net. Проверь файл train_unet.py.")

    model = UNet(n_classes=n_classes)
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
    processor = LayoutLMv3Processor.from_pretrained("ykilcher/layoutlmv3-base-finetuned-publaynet")
    model = LayoutLMv3ForTokenClassification.from_pretrained("ykilcher/layoutlmv3-base-finetuned-publaynet")

    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(device)

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
        # Пропускаем служебные токены
        if token in ["[CLS]", "[SEP]", "[PAD]"] or token.startswith("##"):
            continue

        label = model.config.id2label[label_id]

        x1, y1, x2, y2 = box
        x1 = int(x1 * width / 1000)
        y1 = int(y1 * height / 1000)
        x2 = int(x2 * width / 1000)
        y2 = int(y2 * height / 1000)

        elements.append({
            "token": token,
            "box": [x1, y1, x2, y2],
            "label": label
        })

    print(f"🔍 Найдено элементов: {len(elements)}")  # Должно быть > 0

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
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
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
    print("🚀 Начало обработки PDF-файла...")

    # Шаг 1: Конвертация всех сборников в изображения
    for pdf_file in PDF_FILES:
        base_name = os.path.splitext(pdf_file)[0]
        image_subdir = os.path.join(OUTPUT_IMAGES_ROOT, base_name)
        enhanced_subdir = os.path.join(ENHANCED_IMAGES_ROOT, base_name)

        os.makedirs(image_subdir, exist_ok=True)
        os.makedirs(enhanced_subdir, exist_ok=True)

        pdf_to_images(pdf_file, image_subdir)

        # Шаг 2: Улучшение качества изображений
        parallel_enhance_images(image_subdir, enhanced_subdir, max_workers=4)

    all_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = load_unet_model().to(device)

    # Шаг 3: Обработка всех страниц из всех сборников
    for base_name in [os.path.splitext(pdf)[0] for pdf in PDF_FILES]:
        enhanced_subdir = os.path.join(ENHANCED_IMAGES_ROOT, base_name)
        mask_subdir = os.path.join(SEGMENTED_MASKS_DIR, base_name)
        os.makedirs(mask_subdir, exist_ok=True)

        for filename in os.listdir(enhanced_subdir):
            if filename.startswith("enhanced_page_") and filename.endswith(".png"):
                page_num = int(re.search(r'_(\d+)\.png', filename).group(1))
                enhanced_image_path = os.path.join(enhanced_subdir, filename)
                mask_path = os.path.join(mask_subdir, f"mask_page_{page_num}.png")

                # Шаг 4: Сегментация с использованием U-Net
                mask = segment_page_with_unet(enhanced_image_path, unet_model, mask_path)
                image = Image.open(enhanced_image_path)

                # Шаг 5: OCR по регионам
                segmented_data = ocr_by_class(image, mask)

                # Шаг 6: Анализ layout
                try:
                    layout_elements = detect_layout_elements(enhanced_image_path)
                except Exception as e:
                    print(f"❌ Ошибка анализа layout: {e}")
                    layout_elements = []

                # Шаг 7: Построение графовой модели
                try:
                    graph = build_graph(layout_elements)
                    graph_data = nx.node_link_data(graph)
                except:
                    graph_data = {"nodes": [], "links": []}

                # Шаг 8: Сохранение данных этой страницы
                title = segmented_data.get("title", "")
                authors = segmented_data.get("author", "").splitlines()
                references = segmented_data.get("references", "")

                all_results.append({
                    "file": base_name,
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