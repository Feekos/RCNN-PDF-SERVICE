import os
import json
import cv2
import subprocess
import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from unet_model import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Проверка и генерация аннотаций ---
ANNOTATION_DIR = "annotations"

os.makedirs(ANNOTATION_DIR, exist_ok=True)

# Проверяем, есть ли хотя бы один .json файл в папке annotations
annotation_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.json')]
if not annotation_files:
    print("⚠️ Директория с аннотациями пуста. Запуск generate_annotations.py...")
    try:
        result = subprocess.run(
            ["python", "generate_annotations.py"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✅ Аннотации успешно сгенерированы.")
        print(result.stdout)

        # После генерации повторная проверка
        annotation_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.json')]
        if not annotation_files:
            raise RuntimeError("❌ generate_annotations.py не создал ни одной аннотации. Проверьте логи выше.")

    except subprocess.CalledProcessError as e:
        print("❌ Ошибка при выполнении generate_annotations.py:")
        print(e.stderr)
        raise RuntimeError(f"Не удалось сгенерировать аннотации. Подробности выше.")
else:
    print(f"📂 Найдено {len(annotation_files)} аннотаций в директории {ANNOTATION_DIR}.")

# --- Конфигурация ---
CLASSES = {
    'background': 0,
    'text': 1,
    'title': 2,
    'author': 3,
    'figure': 4,
    'table': 5,
    'bibliography': 6
}

LABEL_MAP = {
    "Figure": "figure",
    "Table": "table",
    "Text": "text",
    "Title": "title",
    "Author": "author",
    "Bibliography": "bibliography"
}

CLASS_NAMES = list(CLASSES.keys())

# --- Датасет ---
class PDFSegmentationDataset(Dataset):
    def __init__(self, image_root, annotation_dir, transform=None):
        self.image_root = image_root
        self.annotation_dir = annotation_dir
        self.transform = transform

        # Собираем все изображения и аннотации
        self.image_paths = []
        self.annotation_paths = []

        for root, _, files in os.walk(image_root):
            for file in files:
                if file.endswith(".png"):
                    full_image_path = os.path.join(root, file)
                    self.image_paths.append(full_image_path)

                    # Получаем имя подпапки и номер страницы
                    subfolder = os.path.basename(root)
                    page_number = file.replace("enhanced_page_", "").replace(".png", "")
                    ann_name = f"{subfolder}_enhanced_page_{page_number}.json"

                    self.annotation_paths.append(os.path.join(annotation_dir, ann_name))

        if len(self.image_paths) != len(self.annotation_paths):
            raise RuntimeError("❌ Количество изображений и аннотаций не совпадает. Проверьте структуру данных.")

        print(f"✅ Найдено {len(self.image_paths)} изображений и соответствующих аннотаций.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]

        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        # Загрузка изображения
        image = Image.open(img_path).convert('RGB')

        # Загрузка аннотаций
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # Парсинг Label Studio JSON
        for pred in annotation.get("predictions", []):
            for res in pred.get("result", []):
                value = res.get("value", {})
                if "rectanglelabels" not in value:
                    continue
                label = value["rectanglelabels"][0]
                x = value["x"] / 100
                y = value["y"] / 100
                width = value["width"] / 100
                height = value["height"] / 100

                # Маппинг меток
                label = LABEL_MAP.get(label, label)
                if label not in CLASSES:
                    continue

                class_id = CLASSES[label]

                x1 = int(x * image.width)
                y1 = int(y * image.height)
                x2 = int((x + width) * image.width)
                y2 = int((y + height) * image.height)

                cv2.rectangle(mask, (x1, y1), (x2, y2), class_id, -1)

        # Применение трансформаций
        if self.transform:
            image = self.transform(image)

        # Ресайз маски
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.tensor(mask_resized, dtype=torch.long)

        return image, mask_tensor


# --- Метрики ---
def compute_metrics(preds, targets, num_classes):
    preds = preds.view(-1)
    targets = targets.view(-1)

    ious = []
    accuracies = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        iou = intersection / (union + 1e-8) if union > 0 else 1.0
        acc = (pred_mask == target_mask).sum().float() / len(targets)

        ious.append(iou.item())
        accuracies.append(acc.item())

    mean_iou = np.mean(ious)
    mean_acc = np.mean(accuracies)

    return mean_iou, mean_acc, ious, accuracies


# --- Визуализация ---
def visualize_prediction(model, dataset, device, idx=0):
    model.eval()
    image, mask = dataset[idx]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    image_np = image_np.astype(np.uint8)

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    print("Unique values in prediction:", np.unique(pred))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(mask.cpu().numpy(), cmap='jet', vmin=0, vmax=len(CLASSES)-1)
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap='jet', vmin=0, vmax=len(CLASSES)-1)
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_result.png")
    plt.show()


# --- Обучение ---
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PDFSegmentationDataset('enhanced_images', 'annotations', transform=transform)

    # Разделение на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = UNet(n_classes=len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    best_val_iou = 0.0
    early_stop_counter = 0
    patience = 5  # Количество эпох без улучшения перед остановкой

    start_time = time.time()

    # === ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ ===
    for epoch in range(100):  # Увеличиваем число эпох для возможности ранней остановки
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"\nEpoch [{epoch + 1}/100], Train Loss: {avg_loss:.4f}")

        # === ВАЛИДАЦИЯ ===
        model.eval()
        total_val_iou = 0
        total_val_acc = 0
        total_samples = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                iou, acc, _, _ = compute_metrics(preds, masks, len(CLASSES))
                total_val_iou += iou * images.size(0)
                total_val_acc += acc * images.size(0)
                total_samples += images.size(0)

        avg_val_iou = total_val_iou / total_samples
        avg_val_acc = total_val_acc / total_samples
        print(f"Validation IoU: {avg_val_iou:.4f}, Accuracy: {avg_val_acc:.4f}")

        # === EARLY STOPPING ===
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            early_stop_counter = 0
            torch.save(model.state_dict(), "unet_publaynet_best.pth")
            print("💾 Сохранена лучшая модель на валидации.")
        else:
            early_stop_counter += 1
            print(f"⚠️ Нет улучшений валидационного IoU уже {early_stop_counter} эпох")

        # Проверка условия остановки
        if early_stop_counter >= patience:
            print(f"\n🛑 Ранняя остановка! Лучший Validation IoU: {best_val_iou:.4f}")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n⏱️ Общее время обучения: {total_time:.2f} секунд")
    print(f"⏱️ Среднее время на эпоху: {total_time / (epoch + 1):.2f} секунд")

    # === СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ ===
    torch.save(model.state_dict(), "unet_publaynet_final.pth")
    print("✅ Финальная модель сохранена")

    # === ГРАФИК ПОТЕРЬ ===
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("График потерь по эпохам для U-Net")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

    # === ВИЗУАЛИЗАЦИЯ ===
    visualize_prediction(model, dataset, device)


if __name__ == "__main__":
    train()