from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Importamos el dataset de prueba
# Usamos el dataset de IMDB
df_reviewsNEW = pd.read_csv("IMDB CleanedDataset.csv")
df_sample = df_reviewsNEW.sample(n=5000, random_state=42)
corpus = df_sample['review']
y = df_sample['sentiment'].map({"negative":0, "positive":1})  # convertimos a 0/1

# Inicializando el vectorizador
vectorizer = CountVectorizer(max_features=5000)  # limitamos el número de palabras para no crear ruido
x = vectorizer.fit_transform(corpus)
X_dense = x.toarray()  # convertimos a array denso para operaciones con numpy

# Separamos los datos para entrenar
xtrain, xtest, ytrain, ytest = train_test_split(X_dense, y, test_size=0.2, random_state=42)

# Equación sigmoide = Esta transforma cualquier número real entre una probabilidad entre 0 y 1
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Bucle de entrenamiento
lr = 0.01  # learning rate
epochs = 1000

# Modelo lineal para las predicciones (z)
n_features = xtrain.shape[1]  # número de columnas
weights = np.zeros(n_features)  # vector de ceros para los pesos
bias = 0

for i in range(epochs):
    z = np.dot(xtrain, weights) + bias
    y_pred = sigmoide(z)

    # Función de coste
    # Calculamos las pérdidas según las probabilidades = penalizamos más cuando nos equivocamos con mucha confianza.
    m = len(ytrain)
    epsilon = 1e-9
    loss = - (1/m) * np.sum(ytrain*np.log(y_pred + epsilon) + (1-ytrain)*np.log(1-y_pred + epsilon))

    # Gradientes: derivadas parciales de la función de coste con respecto a w y b
    dw = (1/m) * np.dot(xtrain.T, (y_pred - ytrain))  # derivada w
    db = (1/m) * np.sum(y_pred - ytrain)  # derivada b

    # Actualizamos los valores
    weights -= lr * dw
    bias -= lr * db

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

# Clasificación final
y_pred_final = sigmoide(np.dot(xtest, weights) + bias)
y_pred_labels = (y_pred_final >= 0.5).astype(int)
accuracy = np.mean(y_pred_labels == ytest)
print("Accuracy:", accuracy)

# Influencia de palabras
feature_names = vectorizer.get_feature_names_out()
coefficients = weights

# Top 20 palabras más positivas
top_positive_indices = np.argsort(coefficients)[-20:]  # ordena de menor a mayor
print("Palabras más positivas:")
for i in top_positive_indices:
    print(feature_names[i], coefficients[i])

# Top 20 palabras más negativas
top_negative_indices = np.argsort(coefficients)[:20]
print("\nPalabras más negativas:")
for i in top_negative_indices:
    print(feature_names[i], coefficients[i])

# Visualización gráfica de los resultados
cm = confusion_matrix(ytest, y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Logistic Regression")
plt.show()
