import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from unet_model import UNet
from tqdm import tqdm

CLASSES = {
    'background': 0,
    'text': 1,
    'title': 2,
    'author': 3,
    'figure': 4,
    'table': 5,
    'bibliography': 6
}

class PDFSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace('.png', '.json'))

        image = Image.open(img_path).convert('RGB')
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        for obj in annotation.get("objects", []):
            label = obj["label"]
            class_id = CLASSES.get(label, 0)
            box = obj["bbox"]
            x1, y1, w, h = map(int, box)
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(mask, (x1, y1), (x2, y2), class_id, -1)

        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PDFSegmentationDataset('images', 'annotations', transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = UNet(n_classes=len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "unet_publaynet.pth")
    print("✅ Модель сохранена")


if __name__ == "__main__":
    train()