import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

# =========================================================================
# 1. CARGAR Y PREPROCESAR
# =========================================================================
df = pd.read_csv("books_google.csv")
df['description'] = df['description'].fillna('')
df = df[df['description'] != ''].copy()

# =========================================================================
# 2. VECTORIZACIÓN (TF-IDF)
# =========================================================================
vectorizer = TfidfVectorizer(max_features=500, stop_words='english', max_df=0.8, min_df=2)
tfidf_matrix = vectorizer.fit_transform(df['description'])

# =========================================================================
# 3. CLUSTERING (K-MEANS) - "Descubriendo Géneros"
# =========================================================================
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# =========================================================================
# 4. CLASIFICACIÓN (Decision Tree) - "Aprendiendo las Reglas"
# =========================================================================
print("\n🌲 Entrenando Clasificador Decision Tree...")
X = tfidf_matrix
y = df['cluster']

# Dividimos para validar qué tan bien aprende el árbol las etiquetas del cluster
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

print(f"Precisión del clasificador: {clf.score(X_test, y_test):.2%}")

# =========================================================================
# 5. VISUALIZACIÓN Y RESULTADOS
# =========================================================================
# Visualizar el Árbol (Lógica de clasificación)
plt.figure(figsize=(20,10))
plot_tree(clf, max_depth=2, feature_names=vectorizer.get_feature_names_out(), 
          class_names=[f"Cluster {i}" for i in range(n_clusters)], filled=True, fontsize=10)
plt.title("Lógica de Clasificación del Árbol de Decisión")
plt.savefig('decision_tree_logic.png')
print("✓ Guardado: decision_tree_logic.png")

# =========================================================================
# 6. FUNCIÓN DE RECOMENDACIÓN Y CLASIFICACIÓN
# =========================================================================
def clasificar_y_recomendar(texto_nuevo, num_recom=3):
    """
    Toma una descripción nueva, la clasifica con el Árbol y recomienda libros
    """
    # Clasificar con el Decision Tree
    vector_nuevo = vectorizer.transform([texto_nuevo])
    cluster_asignado = clf.predict(vector_nuevo)[0]
    
    print(f"\n🔍 Descripción nueva clasificada en: CLUSTER {cluster_asignado}")
    
    # Buscar recomendaciones en ese cluster
    recoms = df[df['cluster'] == cluster_asignado].sort_values(by='rating', ascending=False).head(num_recom)
    
    print("📚 Libros recomendados en esta categoría:")
    for i, row in recoms.iterrows():
        print(f"   - {row['title']} (Rating: {row['rating']})")

# Ejemplo de ejecución
print("\n" + "="*50)
print("PRUEBA DE CLASIFICACIÓN NUEVA")
sinopsis_prueba = "A magic school where young wizards fight against dark lords and dragons."
clasificar_y_recomendar(sinopsis_prueba)