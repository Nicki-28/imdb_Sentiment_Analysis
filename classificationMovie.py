from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import csv

df_reviewsOLD= pd.read_csv ('IMDB Dataset.csv')
db_reviewsNEW= "IMDB CleanedDataset.csv"  # Creamos el nuevo archivo limpio

#limpiamos el texto
   
with open(db_reviewsNEW, mode="w", newline="", encoding="utf-8") as cleaned_file:
    writer = csv.writer(cleaned_file)
    writer.writerow(['review', 'sentiment'])
    for _, row in df_reviewsOLD.iterrows():
        texto = row['review']

        # Limpieza del texto
        texto = re.sub(r'<.*?>', '', texto)
        texto_limpio = re.sub(r'[^A-Za-z0-9\s]', '', texto)  # Quitamos caracteres especiales
        texto_limpio= re.sub(r'\d+', '', texto) #quitamos números
        texto_limpio = texto_limpio.strip().lower()

        # Escribir el texto limpio
        writer.writerow([texto_limpio, row['sentiment']])

df_reviewsNEW = pd.read_csv("IMDB CleanedDataset.csv")

corpus = df_reviewsNEW['review'] #accedo a la columna review del dataframe

#inicializando el vectorizador
vectorizer= CountVectorizer(max_features=5000) # limitamos el numero de palabras para no crear ruido
X= vectorizer.fit_transform(corpus) 

#Vocabulario y frecuencias
#print("Vocabulary:", vectorizer.vocabulary_) # Hacemos un diccionario sobre dónde está cada palabra en la matriz
#print("Feature Names:", vectorizer.get_feature_names_out()) #Palabras del diccionario en orden
#print("Bag of Words Representation:\n", X.toarray())

# Analizar la frecuencias de las palabras
word_counts = np.sum(X.toarray(), axis=0) #suma por columnas para saber cuantas veces se repite cada palabra
word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts)) #emparejamos cada palabra con su frecuencia
#print("Word Frequencies:", word_freq)

#entrenamiento y prueba del modelo
x=X #variables independientes
y= df_reviewsNEW['sentiment'] #variables dependientes

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

model= LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) #precision,etc

#influencia de palabras
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0] #pesos de cada palabra, como es clasificación binaria es solo una fila

#top 20 palabras más positivas
top_positive_indices = np.argsort(coefficients)[-20:] #ordea de mayor a menor
print("Palabras más positivas:")
for i in top_positive_indices:
    print(feature_names[i], coefficients[i])

#top 20 palabras más negativas
top_negative_indices = np.argsort(coefficients)[:20]
print("\nPalabras más negativas:")
for i in top_negative_indices:
    print(feature_names[i], coefficients[i])

#visualización gráfica de los resultados
cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["positive", "negative"], yticklabels=["positive", "negative"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Logistic Regression")
plt.show()