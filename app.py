from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz
import cv2
import numpy as np
import torch
from PIL import Image
from unet_model import UNet
import pytesseract
from torchvision import transforms
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3ForTokenClassification
import networkx as nx
import os

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "unet_publaynet.pth"
IMAGE_SIZE = (512, 512)

CLASSES = {
    0: 'background',
    1: 'text',
    2: 'title',
    3: 'author',
    4: 'figure',
    5: 'table',
    6: 'bibliography'
}

# Загрузка модели
model = UNet(n_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Трансформации
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained("microsoft/layoutlmv3-base")
layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(DEVICE)

# OCR по регионам
def ocr_by_class(image_array, mask_array):
    image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
    result = {}

    class_ids = {2: "title", 3: "author", 6: "references"}
    for cls_id, name in class_ids.items():
        coords = cv2.findNonZero((mask_array == cls_id).astype(np.uint8))
        if coords is None or len(coords) == 0:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        roi = image[y:y+h, x:x+w]
        result[name] = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))).strip()

    return result

@app.post("/process_pdf/")
async def process_pdf(pdf_file: UploadFile = File(...)):
    with open("temp.pdf", "wb") as f:
        f.write(await pdf_file.read())
    doc = fitz.open("temp.pdf")
    results = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        image_path = f"page_{page_num}.jpg"
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image.save(image_path)

        # Предобработка
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # OCR
        segmented_data = ocr_by_class(image, mask)

        # Layout анализ
        try:
            encoding = feature_extractor(image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                layout_out = layout_model(**encoding)
                layout_pred = torch.argmax(layout_out.logits, dim=-1)
            tokens = feature_extractor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            boxes = encoding["bbox"][0].tolist()
            layout_elements = [
                {"token": token, "box": box, "label": layout_model.config.id2label[label.item()]}
                for token, box, label in zip(tokens, boxes, layout_pred[0])
                if token not in ["[CLS]", "[SEP]"] and not token.startswith("##")
            ]
        except Exception as e:
            layout_elements = []

        results.append({
            "page": page_num,
            "title": segmented_data.get("title", ""),
            "authors": segmented_data.get("author", "").splitlines(),
            "bibliography": segmented_data.get("references", ""),
            "layout_elements": layout_elements
        })

    os.remove("temp.pdf")
    return JSONResponse({"results": results})