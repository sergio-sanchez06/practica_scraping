# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics.pairwise import cosine_similarity
# from deep_translator import GoogleTranslator

# # =========================================================================
# # 1. PREPARACIÓN DE DATOS
# # =========================================================================
# print("🚀 Cargando motor de inteligencia bilingüe...")
# try:
#     df = pd.read_csv("books_google.csv")
#     df['description'] = df['description'].fillna('')
#     df = df[df['description'] != ''].copy().reset_index(drop=True)
# except FileNotFoundError:
#     print("❌ Error: No se encontró 'books_google.csv'.")
#     exit()

# # =========================================================================
# # 2. PROCESAMIENTO NLP (TF-IDF)
# # =========================================================================
# # Analizamos palabras sueltas y frases de dos palabras (ej: "magic school")
# vectorizer = TfidfVectorizer(
#     max_features=2500, 
#     stop_words='english',
#     ngram_range=(1, 2)
# )
# tfidf_matrix = vectorizer.fit_transform(df['description'])

# # =========================================================================
# # 3. CLUSTERING Y CLASIFICACIÓN
# # =========================================================================
# n_clusters = 10
# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
# df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# clf = DecisionTreeClassifier(max_depth=20, random_state=42)
# clf.fit(tfidf_matrix, df['cluster'])

# # =========================================================================
# # 4. FUNCIÓN DE RECOMENDACIÓN MEJORADA
# # =========================================================================
# def recomendar_inteligente(texto_usuario, top_n=5):
#     try:
#         # A. Traducción automática
#         traductor = GoogleTranslator(source='auto', target='en')
#         texto_en = traductor.translate(texto_usuario)
#         print(f"\n" + "-"*50)
#         print(f"🔍 Buscando: '{texto_usuario}'")
#         if texto_usuario.lower() != texto_en.lower():
#             print(f"🤖 Traducido como: '{texto_en}'")

#         # B. Clasificación mediante el Árbol de Decisión
#         vec_usuario = vectorizer.transform([texto_en])
#         cluster_id = clf.predict(vec_usuario)[0]
#         print(f"📍 Estante asignado: CLÚSTER {cluster_id}")

#         # C. Similitud del Coseno (Solo dentro del cluster para precisión)
#         indices_cluster = df[df['cluster'] == cluster_id].index
#         matriz_cluster = tfidf_matrix[indices_cluster]
#         similitudes = cosine_similarity(vec_usuario, matriz_cluster).flatten()
        
#         # Tomamos los top_n mejores resultados
#         top_indices_locales = similitudes.argsort()[-top_n:][::-1]
        
#         nombres_recom = []
#         scores_recom = []

#         print(f"📚 Top {top_n} recomendaciones para ti:")
#         print("-" * 50)
        
#         for idx in top_indices_locales:
#             indice_original = indices_cluster[idx]
#             libro = df.iloc[indice_original]
#             score = similitudes[idx]
            
#             # Semáforo de confianza
#             if score > 0.20:
#                 emoji, color = "✅", "green"
#             elif score > 0.10:
#                 emoji, color = "⚠️", "orange"
#             else:
#                 emoji, color = "❓", "red"
            
#             print(f" {emoji} {libro['title']} ({score:.2%})")
#             print(f"    Autor: {libro['author']}")
            
#             nombres_recom.append(libro['title'][:25] + "...")
#             scores_recom.append(score)

#         # D. Generación de Gráfico (Opcional, se puede comentar en entornos sin GUI)
#         generar_grafico(nombres_recom, scores_recom, texto_usuario)
            
#     except Exception as e:
#         print(f"❌ Error en la recomendación: {e}")

# def generar_grafico(nombres, scores, busqueda):
#     plt.figure(figsize=(10, 5))
#     colors = ['green' if s > 0.20 else 'orange' if s > 0.10 else 'red' for s in scores]
#     plt.barh(nombres, scores, color=colors)
#     plt.gca().invert_yaxis()
#     plt.xlabel('Nivel de Similitud (Coseno)')
#     plt.title(f'Resultados para: "{busqueda}"')
#     plt.tight_layout()
#     plt.show()

# # =========================================================================
# # 5. INTERFAZ DE USUARIO
# # =========================================================================


# print("\n" + "="*60)
# print("📚 RECOMENDADOR DAW V3.0 (Coseno + Árbol + Traductor)")
# print("Escribe tu interés (ej: 'Misterios en Londres' o 'Dragons')")
# print("Escribe 'salir' para terminar.")
# print("="*60)

# while True:
#     entrada = input("\n📝 ¿Qué buscas?: ")
#     if entrada.lower() in ['salir', 'exit', 'quit']:
#         print("👋 ¡Hasta luego!")
#         break
#     if len(entrada.strip()) < 3:
#         print("⚠️ Escribe algo más descriptivo.")
#         continue
#     recomendar_inteligente(entrada)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re  # Para detectar números en el texto
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# =========================================================================
# 1. PREPARACIÓN Y CONFIGURACIÓN
# =========================================================================
print("🚀 Cargando motor inteligente V4.0...")
try:
    df = pd.read_csv("books_google.csv")
    df['description'] = df['description'].fillna('')
    df = df[df['description'] != ''].copy().reset_index(drop=True)
except FileNotFoundError:
    print("❌ Error: No se encontró 'books_google.csv'.")
    exit()

vectorizer = TfidfVectorizer(max_features=2500, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['description'])

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

clf = DecisionTreeClassifier(max_depth=20, random_state=42)
clf.fit(tfidf_matrix, df['cluster'])

# =========================================================================
# 2. FUNCIÓN DE EXTRACCIÓN DE CANTIDAD
# =========================================================================
def extraer_cantidad(texto):
    numeros = re.findall(r'\d+', texto)
    if numeros:
        cantidad = int(numeros[0])
        return max(1, min(cantidad, 20)) # Límite entre 1 y 20 para no saturar
    return 5 # Por defecto

# =========================================================================
# 3. RECOMENDADOR CON DETECCIÓN DE INTENCIÓN
# =========================================================================
def recomendar_inteligente(texto_usuario):
    try:
        # Detectar cuántos libros quiere el usuario
        limite = extraer_cantidad(texto_usuario)
        
        # Traducción
        traductor = GoogleTranslator(source='auto', target='en')
        texto_en = traductor.translate(texto_usuario)
        
        print(f"\n" + "="*50)
        print(f"🔍 Búsqueda: '{texto_usuario}'")
        print(f"📊 Solicitados: {limite} resultados")

        # Clasificación
        vec_usuario = vectorizer.transform([texto_en])
        cluster_id = clf.predict(vec_usuario)[0]

        # Similitud (Coseno)
        indices_cluster = df[df['cluster'] == cluster_id].index
        matriz_cluster = tfidf_matrix[indices_cluster]
        similitudes = cosine_similarity(vec_usuario, matriz_cluster).flatten()
        
        # Ajustar límite si el cluster tiene pocos libros
        ajuste_limite = min(limite, len(indices_cluster))
        top_indices_locales = similitudes.argsort()[-ajuste_limite:][::-1]
        
        nombres_recom = []
        scores_recom = []

        print(f"📚 Recomendaciones encontradas:")
        for idx in top_indices_locales:
            indice_original = indices_cluster[idx]
            libro = df.iloc[indice_original]
            score = similitudes[idx]
            
            emoji = "✅" if score > 0.15 else "❓"
            print(f" {emoji} {libro['title']} ({score:.2%})")
            
            nombres_recom.append(libro['title'][:25] + "...")
            scores_recom.append(score)

        # Gráfico
        if nombres_recom:
            plt.figure(figsize=(10, 5))
            plt.barh(nombres_recom, scores_recom, color='skyblue')
            plt.gca().invert_yaxis()
            plt.title(f'Similitud para: {texto_en[:40]}...')
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")

# =========================================================================
# 4. BUCLE PRINCIPAL
# =========================================================================
print("\n" + "="*60)
print("📚 BIENVENIDO AL RECOMENDADOR INTELIGENTE")
print("Ejemplo: 'Dime 8 libros de ciencia ficcion'")
print("="*60)

while True:
    entrada = input("\n📝 ¿Qué te apetece?: ")
    if entrada.lower() in ['salir', 'exit', 'quit']: break
    recomendar_inteligente(entrada)