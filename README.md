# 🎬 IMDB Sentiment Analysis

## 📌 Descripción
Este proyecto implementa un clasificador de **análisis de sentimientos** para reseñas de películas del dataset IMDB, utilizando **Bag of Words** y un modelo de **Regresión Logística**.

El objetivo es clasificar reseñas como **positivas** o **negativas**, practicando el flujo completo de un proyecto de Machine Learning supervisado:
- Preprocesamiento y limpieza de texto
- Vectorización con Bag of Words
- Entrenamiento y evaluación de un modelo
- Interpretación de resultados

---

## 📂 Dataset
El dataset proviene de [Kaggle - IMDB Dataset of 50K Movie Reviews]([https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data]).

⚠️ **Nota:**  
Los archivos CSV no se incluyen en este repositorio debido a su tamaño.  
Para reproducir este proyecto:  
1. Descarga el dataset desde el enlace de Kaggle.  
2. Coloca el archivo `IMDB Dataset.csv` en la carpeta raíz del proyecto.

---

## 🚀 Flujo del proyecto
1. **Carga y limpieza de datos**  
2. **Vectorización (Bag of Words)**  
3. **División Train/Test**  
4. **Entrenamiento**  
5. **Evaluación**  
6. **Interpretación**  
---

## 📊 Resultados
- **Accuracy:** ~0.88  
- **Palabras más influyentes:**  
  - *Positivas:* excellent, amazing, wonderful, ...  
  - *Negativas:* boring, worst, waste, ...

---

## 📦 Instalación y ejecución
1. Clonar repositorio:  
   ```bash
   git clone https://github.com/Nicki-28/imdb_Sentiment_Analysis.git
   cd imdb_Sentiment_Analysis
2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
