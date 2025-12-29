# CLUSTERING DE LIBROS - EXPLICACIÓN DETALLADA

## Objetivo del Proyecto

Agrupar automáticamente 300 libros de Goodreads en categorías similares usando **Machine Learning no supervisado**.

---

## ¿Qué es el Clustering?

El **clustering** es una técnica de Machine Learning que agrupa elementos similares sin necesidad de etiquetas previas. Es como cuando organizas tu biblioteca: pones los libros de terror juntos, los de romance juntos, etc., pero lo hace la máquina automáticamente.

### Diferencia vs Clasificación:

- **Clasificación** (supervisado): Necesitas ejemplos etiquetados → "Este libro es de terror, este de romance"
- **Clustering** (no supervisado): La máquina encuentra grupos por sí misma → "Estos libros se parecen entre sí"

---

## Tecnologías Utilizadas

### 1. **TF-IDF** (Term Frequency-Inverse Document Frequency)

Convierte texto en números que las máquinas pueden procesar.

#### ¿Cómo funciona?

**TF (Term Frequency)**: Frecuencia del término en el documento

```
TF = (Número de veces que aparece la palabra) / (Total de palabras en el documento)
```

**IDF (Inverse Document Frequency)**: Penaliza palabras muy comunes

```
IDF = log(Total de documentos / Documentos que contienen la palabra)
```

**TF-IDF final**:

```
TF-IDF = TF × IDF
```

#### Ejemplo Práctico:

Descripción de "The Hunger Games":

```
"Katniss Everdeen participates in the Hunger Games, a fight to the death..."
```

| Palabra | TF   | IDF  | TF-IDF                          |
| ------- | ---- | ---- | ------------------------------- |
| Katniss | 0.05 | 5.2  | **0.26** (importante)           |
| Games   | 0.03 | 3.1  | **0.09** (importante)           |
| the     | 0.15 | 0.1  | **0.015** (poco importante)     |
| a       | 0.10 | 0.05 | **0.005** (muy poco importante) |

**Resultado**: Las palabras específicas como "Katniss" y "Games" tienen más peso que palabras comunes como "the" o "a".

---

### 2. **K-Means Clustering**

Algoritmo que agrupa datos en K clusters (grupos).

#### Proceso paso a paso:

```
1. INICIALIZACIÓN
   - Decides K = 6 clusters
   - Se colocan 6 centroides aleatorios

2. ASIGNACIÓN
   - Cada libro se asigna al centroide más cercano

3. ACTUALIZACIÓN
   - Se recalcula la posición de cada centroide (promedio de sus libros)

4. REPETIR
   - Pasos 2-3 hasta que no haya cambios
```

#### Visualización:

```
Iteración 0 (inicio):
  ●     ○
    ○     ●
  ○   ●
    ○     ○
  ●     ○

Iteración 5 (convergencia):
  ● ● ●

  ○ ○ ○

  ◆ ◆ ◆
```

---

## Estructura del Código

### Paso 1: Cargar Datos

```python
df = pd.read_csv("books_google.csv")
# title, author, rating, genres, description
```

### Paso 2: Preprocesar

```python
# Eliminar libros sin descripción
df['description'] = df['description'].fillna('')
df = df[df['description'] != '']
```

### Paso 3: TF-IDF

```python
vectorizer = TfidfVectorizer(
    max_features=500,      # Considerar top 500 palabras
    max_df=0.8,           # Ignorar si aparece en >80% libros
    min_df=2,             # Ignorar si aparece en <2 libros
    stop_words='english'  # Eliminar "the", "a", "is"...
)

tfidf_matrix = vectorizer.fit_transform(df['description'])
# Resultado: matriz de 300 libros × 500 palabras
```

### Paso 4: K-Means

```python
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)
```

### Paso 5: Analizar Resultados

```python
# Palabras clave por cluster
for cluster in range(6):
    centroid = kmeans.cluster_centers_[cluster]
    top_words = get_top_words(centroid)
    print(f"Cluster {cluster}: {top_words}")
```

---

## Resultados Esperados

### Ejemplo de Clusters:

**Cluster 0: Fantasía Juvenil**

- The Hunger Games
- Divergent
- Harry Potter
- Palabras clave: _magic, world, young, adventure_

**Cluster 1: Clásicos**

- Pride and Prejudice
- Jane Eyre
- Great Expectations
- Palabras clave: _love, society, family, life_

**Cluster 2: Thrillers**

- Gone Girl
- The Girl on the Train
- Shutter Island
- Palabras clave: _murder, mystery, detective, truth_

**Cluster 3: Ciencia Ficción**

- 1984
- Brave New World
- Fahrenheit 451
- Palabras clave: _future, society, world, control_

**Cluster 4: Romance/Drama**

- The Notebook
- The Fault in Our Stars
- Me Before You
- Palabras clave: _love, life, heart, story_

**Cluster 5: Horror/Terror**

- It
- The Shining
- Dracula
- Palabras clave: _dark, fear, night, death_

---

## Visualizaciones

### 1. **Scatter Plot 2D**

```
Usa PCA (Principal Component Analysis) para reducir
500 dimensiones → 2 dimensiones visualizables

Cada punto = un libro
Color = cluster asignado
```

### 2. **Gráfico de Barras**

```
Muestra cuántos libros hay en cada cluster
```

---

## 💡 Sistema de Recomendación

Una vez tenemos los clusters, podemos recomendar libros:

```python
def recomendar_libros(titulo):
    # 1. Encuentra el libro
    libro = buscar(titulo)

    # 2. Obtén su cluster
    cluster = libro['cluster']

    # 3. Recomienda otros del mismo cluster
    recomendaciones = libros_en_cluster(cluster)

    return recomendaciones
```

**Ejemplo**:

```
Usuario lee: "Harry Potter"
Cluster: 0 (Fantasía Juvenil)
Recomendaciones:
  • The Hunger Games
  • Percy Jackson
  • Divergent
  • Eragon
```

---

## Cómo Ejecutar

### Instalación de dependencias:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Ejecutar el script:

```bash
python clustering_books.py
```

### Archivos generados:

```
✓ books_clustered.csv         → Datos con cluster asignado
✓ cluster_summary.csv          → Estadísticas por cluster
✓ clusters_visualization.png   → Visualización 2D
✓ cluster_distribution.png     → Distribución de libros
```

---

## Conceptos Clave para el Examen

### 1. **TF-IDF**

- Convierte texto → números
- Da más importancia a palabras específicas
- Reduce importancia de palabras comunes

### 2. **K-Means**

- Algoritmo de clustering
- Requiere definir K (número de clusters)
- Minimiza distancia dentro de clusters

### 3. **Aplicación**

- Recomendación de productos
- Segmentación de clientes
- Organización de documentos
- Detección de patrones

---

## Alternativa: DecisionTreeClassifier

El enunciado también menciona clasificación de género. Aquí la diferencia:

### Clustering (K-Means):

```python
# No necesitas etiquetas
kmeans = KMeans(n_clusters=6)
clusters = kmeans.fit_predict(tfidf_matrix)
```

### Clasificación (DecisionTree):

```python
# SÍ necesitas etiquetas (géneros conocidos)
X = tfidf_matrix
y = df['genres']  # Etiquetas conocidas

clf = DecisionTreeClassifier()
clf.fit(X, y)  # Entrenar
predicciones = clf.predict(X_nuevo)  # Predecir nuevos
```

**Para tu proyecto, K-Means es más apropiado** porque:

1. No tienes géneros limpios para todos los libros
2. Los géneros de Google Books son inconsistentes
3. El clustering descubrirá grupos naturales

---

## Preguntas Frecuentes

**P: ¿Por qué 6 clusters?**
R: Es un balance. Puedes probar con 4-10 y elegir el mejor usando el "método del codo" (elbow method).

**P: ¿Qué pasa si un libro no tiene descripción?**
R: Se elimina del análisis (no se puede clusterizar sin texto).

**P: ¿Los clusters tienen nombres?**
R: No, K-Means solo da números (0, 1, 2...). Tú les pones nombres analizando las palabras clave.

**P: ¿Es mejor que clasificación?**
R: Depende. Clustering descubre patrones, clasificación necesita ejemplos previos.

---

## 📖 Recursos Adicionales

- [Documentación scikit-learn K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [TF-IDF explicado](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Visualización con PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---
