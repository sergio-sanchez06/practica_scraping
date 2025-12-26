import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
#   CARGAR DATOS
# =========================================================================
print("Cargando datos de libros")
df = pd.read_csv("books_google.csv")
print(f"Total de libros cargados: {len(df)}")

# =========================================================================
#   PREPROCESAR DATOS
# =========================================================================
print("Preprocesando descripciones")

# Limpiar descripciones nulas o vacías
df['description'] = df['description'].fillna('')
df = df[df['description'] != '']  # Eliminar libros sin descripción

print(f"Libros con descripción válida: {len(df)}")

# =========================================================================
#   APLICAR TF-IDF
# =========================================================================
print("Aplicando TF-IDF a las descripciones")

# Configurar vectorizador TF-IDF
# - max_features: número máximo de palabras a considerar
# - max_df: ignora palabras que aparecen en más del 80% de documentos
# - min_df: ignora palabras que aparecen en menos de 2 documentos
# - stop_words: elimina palabras comunes en inglés
vectorizer = TfidfVectorizer(
    max_features=500,
    max_df=0.8,
    min_df=2,
    stop_words='english'
)

# Transformar descripciones a matriz TF-IDF
tfidf_matrix = vectorizer.fit_transform(df['description'])
print(f"   Dimensiones de la matriz TF-IDF: {tfidf_matrix.shape}")
print(f"   (libros x características)")

# =========================================================================
#   APLICAR K-MEANS
# =========================================================================
print("\n🎯 Aplicando K-Means clustering...")

# Número de clusters (grupos)
n_clusters = 6

# Crear modelo K-Means
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    max_iter=300,
    n_init=10
)

# Entrenar modelo y predecir clusters
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

print(f"Libros agrupados en {n_clusters} clusters")

# =========================================================================
#   ANÁLISIS DE RESULTADOS
# =========================================================================
print("Análisis de clusters")
print("=" * 70)

for cluster_num in range(n_clusters):
    cluster_books = df[df['cluster'] == cluster_num]
    print(f"\n🏷️  CLUSTER {cluster_num} ({len(cluster_books)} libros)")
    print("-" * 70)
    
    # Mostrar algunos títulos representativos
    print("   Ejemplos de libros:")
    for idx, row in cluster_books.head(5).iterrows():
        print(f"      • {row['title']}")
    
    # Mostrar géneros más comunes
    if cluster_books['genres'].notna().any():
        all_genres = []
        for genres in cluster_books['genres'].dropna():
            if isinstance(genres, str) and genres.startswith('['):
                import ast
                try:
                    genre_list = ast.literal_eval(genres)
                    all_genres.extend(genre_list)
                except:
                    pass
        
        if all_genres:
            from collections import Counter
            genre_counts = Counter(all_genres)
            top_genres = genre_counts.most_common(3)
            print("   Géneros dominantes:")
            for genre, count in top_genres:
                print(f"      • {genre} ({count} libros)")

# =========================================================================
#   PALABRAS CLAVE POR CLUSTER
# =========================================================================
print("\n\nPalabras clave por cluster")
print("=" * 70)

# Obtener nombres de características (palabras)
feature_names = vectorizer.get_feature_names_out()

# Para cada cluster, encontrar las palabras más representativas
for cluster_num in range(n_clusters):
    print(f"\nCLUSTER {cluster_num}:")
    
    # Obtener el centroide del cluster
    centroid = kmeans.cluster_centers_[cluster_num]
    
    # Ordenar palabras por importancia
    top_indices = centroid.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    
    print(f"   Palabras: {', '.join(top_words)}")

# =========================================================================
#   GUARDAR RESULTADOS
# =========================================================================
print("\n\nGuardando resultados")

# Guardar CSV con clusters
df.to_csv("books_clustered.csv", index=False, encoding='utf-8')
print("Guardado: books_clustered.csv")

# Crear resumen de clusters
cluster_summary = df.groupby('cluster').agg({
    'title': 'count',
    'rating': 'mean'
}).rename(columns={'title': 'num_books', 'rating': 'avg_rating'})

cluster_summary.to_csv("cluster_summary.csv", encoding='utf-8')
print("Guardado: cluster_summary.csv")

# =========================================================================
#   VISUALIZACIÓN
# =========================================================================
print("\nGenerando visualizaciones")

# Reducir dimensionalidad para visualizar (PCA)
pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(tfidf_matrix.toarray())

# Crear DataFrame para visualización
viz_df = pd.DataFrame({
    'x': coords_2d[:, 0],
    'y': coords_2d[:, 1],
    'cluster': df['cluster'],
    'title': df['title']
})

# Crear gráfico de dispersión
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    viz_df['x'],
    viz_df['y'],
    c=viz_df['cluster'],
    cmap='viridis',
    alpha=0.6,
    s=100
)

plt.colorbar(scatter, label='Cluster')
plt.title('Clustering de Libros de Goodreads\n(TF-IDF + K-Means)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('clusters_visualization.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: clusters_visualization.png")

# Gráfico de barras: número de libros por cluster
plt.figure(figsize=(10, 6))
cluster_counts = df['cluster'].value_counts().sort_index()
plt.bar(range(n_clusters), cluster_counts, color='steelblue', alpha=0.7)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Número de Libros', fontsize=12)
plt.title('Distribución de Libros por Cluster', fontsize=14, fontweight='bold')
plt.xticks(range(n_clusters))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: cluster_distribution.png")

# =========================================================================
#   SISTEMA DE RECOMENDACIÓN
# =========================================================================
print("\n\nEjemplo de Sistema de Recomendación")
print("=" * 70)

def recomendar_libros(titulo, num_recomendaciones=5):
    """
    Recomienda libros similares basados en el cluster
    """
    # Buscar el libro
    libro = df[df['title'].str.contains(titulo, case=False, na=False)]
    
    if libro.empty:
        print(f"No se encontró el libro '{titulo}'")
        return
    
    libro = libro.iloc[0]
    cluster_libro = libro['cluster']
    
    print(f"Libro seleccionado: {libro['title']}\n")
    print(f"Cluster: {cluster_libro}\n")
    print(f"Recomendaciones (del mismo cluster):\n")
    
    # Obtener libros del mismo cluster
    recomendaciones = df[
        (df['cluster'] == cluster_libro) & 
        (df['title'] != libro['title'])
    ].head(num_recomendaciones)
    
    for idx, rec in enumerate(recomendaciones.iterrows(), 1):
        row = rec[1]
        print(f"   {idx}. {row['title']}")
        print(f"      Autor: {row['author']}")
        print(f"      Rating: {row['rating']}\n")

# Probar con algunos libros
recomendar_libros("Harry Potter", num_recomendaciones=5)
recomendar_libros("1984", num_recomendaciones=5)

print("\n" + "=" * 70)
print("PROCESO DE CLUSTERING COMPLETADO")
print("=" * 70)
print("\nArchivos generados:")
print("books_clustered.csv - Datos con clusters asignados")
print("cluster_summary.csv - Resumen estadístico")
print("clusters_visualization.png - Visualización 2D")
print("cluster_distribution.png - Distribución de libros")
print("\n" + "=" * 70)
