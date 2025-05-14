### 📂 Файловая струкртура проекта

Root /<br>
├── app.py                  ← FastAPI API <br>
├── streamlit_app.py        ← Веб-интерфейс на Streamlit UI <br>
├── unet_model.py           ← U-Net модель <br>
├── main.py                 ← основной пайплайн процесса подготовки и анализа данных <br>
├── train_unet.py           ← обучение модели <br>
├── generate_annotations.py ← автоматическая разметка <br>
├── requirements.txt <br>
├── Dockerfile <br>
├── docker-compose.yml      ← объединяет FastAPI + Streamlit <br>
├── sbornik_N.pdf           ← сборник(-и) статей в формате PDF <br>
├── images/                 ← изображения страниц PDF <br>
├── enhanced_images/        ← улучшенные изображения <br>
├── segmented_masks/        ← маски сегментации <br>
└── annotations/            ← аннотации для обучения <br>

### 🧠 Поддерживаемые функции
✅ Конвертация PDF в изображения <br>
✅ Улучшение качества изображений с помощью OpenCV <br>
✅ Сегментация страниц с помощью U-Net <br>
✅ Анализ layout с помощью LayoutLMv3 <br>
✅ OCR по регионам документа<br>
✅ Извлечение метаданных: заголовок, авторы, список литературы <br>
✅ Графовая модель связей между элементами документа <br>
✅ Экспорт обработанных данных в JSON, CSV, XLSX <br>
✅ REST API (FastAPI) <br>
✅ Пользовательский веб-интерфейс (Streamlit) <br>
✅ Автоматическая генерация аннотаций <br>
✅ Интеграция с Label Studio <br>