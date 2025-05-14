import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import networkx as nx
import base64
import tempfile
import pandas as pd

# === Настройки ===
FASTAPI_URL = "http://localhost:8000/process_pdf"

# === Центральная панель ===
st.set_page_config(page_title="PDF Analyser", layout="wide")
st.title("📄 Анализ статей из PDF-документов")
st.markdown("Сервис для обработки публикаций (статей) в виде PDF-документа для получения структурированного анализа с использованием U-Net, LayoutLMv3 и графовой модели.")
st.caption("Разработан в рамках ВКР на тему: \n**Модель нейронных сетей лингвистической обработки данных в корпоративных информационных системах для преобразования файлов в формате PDF в машиночитаемый текст**")

# === Боковая панель ===
st.sidebar.header("Загрузите документ")
uploaded_file = st.sidebar.file_uploader("Выберите PDF-файл", type="pdf")

if uploaded_file is not None:
    # Сохранение временного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        temp_pdf_path = tmpfile.name

    st.sidebar.success("Файл успешно загружен!")

    if st.sidebar.button("🚀 Начать обработку"):
        with st.spinner("Обработка PDF..."):
            files = {"pdf_file": (uploaded_file.name, open(temp_pdf_path, "rb"), "application/pdf")}
            try:
                response = requests.post(FASTAPI_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    results = result.get("results", [])

                    st.sidebar.success("Обработка завершена!")

                    # === Подготовка таблицы ===
                    table_data = []

                    for page_data in results:
                        page_num = page_data["page"]
                        title = page_data["title"] or "Не найдено"
                        authors = ", ".join(page_data["authors"]) if page_data["authors"] else "Не найдены"
                        bibliography = page_data["bibliography"] or "Не найдена"

                        table_data.append({
                            "Страница": page_num + 1,
                            "Название статьи": title,
                            "Авторы": authors,
                            "Литература": bibliography
                        })

                    df = pd.DataFrame(table_data)

                    # === Кнопки для скачивания CSV и XLSX ===
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Скачать как CSV",
                            data=df.to_csv(index=False, sep=";", encoding="utf-8-sig"),
                            file_name="structured_output.csv",
                            mime="text/csv"
                        )
                    with col2:
                        towrite = io.BytesIO()
                        df.to_excel(towrite, index=False, engine='openpyxl')
                        towrite.seek(0)  # rewind the buffer
                        st.download_button(
                            label="📥 Скачать как XLSX",
                            data=towrite,
                            file_name="structured_output.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    # === Отображение результатов постранично ===
                    for page_data in results:
                        page_num = page_data["page"]
                        title = page_data["title"]
                        authors = page_data["authors"]
                        bibliography = page_data["bibliography"]

                        st.subheader(f"Страница {page_num + 1}")
                        st.markdown(f"**### 🔖 Название статьи**\n{title or 'Не найдено'}")
                        st.markdown(f"**### 👤 Авторы**\n{', '.join(authors) if authors else 'Не найдены'}")
                        st.markdown(f"**### 📚 Литература**\n{bibliography or 'Не найдена'}")

                        # Отображение маски
                        mask_array = np.array(page_data["mask_preview"])
                        mask_image = Image.fromarray((mask_array * 255 / mask_array.max()).astype(np.uint8))
                        st.image(mask_image, caption=f"Маска сегментации — страница {page_num + 1}", use_column_width=True)

                        # Графовая модель
                        try:
                            G = nx.node_link_graph(page_data["graph"])
                            plt.figure(figsize=(10, 6))
                            nx.draw(G, with_labels=True, node_size=700, node_color="lightblue")
                            st.pyplot(plt)
                        except Exception as e:
                            st.warning("⚠️ Не удалось отобразить граф для этой страницы.")

                else:
                    st.error("Ошибка API: код ответа {}".format(response.status_code))
                    st.code(response.text)

            except Exception as e:
                st.error(f"⚠️ Произошла ошибка:\n{str(e)}")

else:
    st.info("Пожалуйста, загрузите PDF-файл для начала работы.")