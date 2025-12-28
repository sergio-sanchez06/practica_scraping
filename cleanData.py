import pandas as pd
import json
import re
import os

# --- 1. FUNCIONES DE LIMPIEZA AVANZADA (PREPARACIÓN PARA IA) ---

def limpiar_texto_nlp(texto):
    """
    Normaliza texto para entrenamiento de IA:
    - Minúsculas.
    - Elimina HTML (<br>, <i>).
    - Elimina puntuación irrelevante.
    """
    if not isinstance(texto, str):
        return ""
    
    # Decodificar caracteres unicode o html si existen
    texto = texto.lower()
    # Eliminar etiquetas HTML
    texto = re.sub(r'<.*?>', ' ', texto)
    # Eliminar caracteres que no sean alfanuméricos básicos (preservando espacios)
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', '', texto)
    # Colapsar múltiples espacios en uno solo
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def normalizar_rating(valor):
    """Convierte '4.35 avg rating' a float 4.35"""
    try:
        if pd.isna(valor): return 0.0
        # Buscar patrón numérico (ej. 4.5 o 4.50)
        match = re.search(r"(\d+\.\d+)", str(valor))
        return float(match.group(1)) if match else 0.0
    except:
        return 0.0

def normalizar_generos(valor):
    """
    Estandariza los géneros.
    - JSON suele traer listas: ['Fiction']
    - CSV suele traer strings que parecen listas: "['Fiction']"
    Retorna string separado por espacios para vectorización.
    """
    if pd.isna(valor): return ""
    
    # Si ya es lista (viene del JSON)
    if isinstance(valor, list):
        return " ".join([limpiar_texto_nlp(g) for g in valor])
    
    # Si es string (viene del CSV o JSON malformado)
    if isinstance(valor, str):
        # Limpiar corchetes y comillas sobrantes del formato string de lista
        limpio = re.sub(r"[\[\]'\",]", " ", valor)
        return limpiar_texto_nlp(limpio)
    
    return ""

# --- 2. CARGA Y FUSIÓN DE DATOS ---

def generar_dataset_unificado(ruta_csv, ruta_json):
    print("Cargando archivos...")
    
    # Cargar CSV
    try:
        df_csv = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print("Advertencia: No se encontró el CSV.")
        df_csv = pd.DataFrame()

    # Cargar JSON
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        df_json = pd.DataFrame(data_json)
    except FileNotFoundError:
        print("Advertencia: No se encontró el JSON.")
        df_json = pd.DataFrame()

    # Fusionar (Concatenar)
    print("Fusionando datasets...")
    df_full = pd.concat([df_csv, df_json], ignore_index=True)
    
    # ELIMINAR DUPLICADOS
    # Es vital para no sesgar a la IA con el mismo libro repetido.
    # Usamos Título y Autor como clave única.
    total_inicial = len(df_full)
    df_full.drop_duplicates(subset=['title', 'author'], keep='first', inplace=True)
    print(f"Registros iniciales: {total_inicial} -> Registros únicos: {len(df_full)}")

    # --- 3. APLICAR LIMPIEZA ---
    print("Limpiando datos para IA...")
    
    # Copia para no alterar estructura original drásticamente si no se desea,
    # pero aquí sobrescribimos para el formato final de entrenamiento.
    df_full['title_clean'] = df_full['title'].apply(limpiar_texto_nlp)
    df_full['author_clean'] = df_full['author'].apply(limpiar_texto_nlp)
    df_full['genres_clean'] = df_full['genres'].apply(normalizar_generos)
    df_full['description_clean'] = df_full['description'].apply(limpiar_texto_nlp)
    df_full['rating_num'] = df_full['rating'].apply(normalizar_rating)
    
    # Rellenar nulos restantes
    df_full.fillna('', inplace=True)
    
    return df_full

def filtrar_dataset(df, genero=None, autor=None, palabra_clave=None):
    """
    Filtra sobre las columnas limpias (_clean) para mayor precisión.
    """
    mask = pd.Series([True] * len(df))
    
    if genero:
        mask &= df['genres_clean'].str.contains(limpiar_texto_nlp(genero), na=False)
    if autor:
        mask &= df['author_clean'].str.contains(limpiar_texto_nlp(autor), na=False)
    if palabra_clave:
        kw = limpiar_texto_nlp(palabra_clave)
        mask &= (df['description_clean'].str.contains(kw, na=False) | 
                 df['title_clean'].str.contains(kw, na=False))
                 
    return df[mask]

# --- EJECUCIÓN PRINCIPAL ---

archivo_csv = 'books_google.csv'
archivo_json = 'books_google.json'

# 1. Generar el dataset maestro
df_maestro = generar_dataset_unificado(archivo_csv, archivo_json)

# 2. Guardar dataset procesado (Listo para TensorFlow/PyTorch/Scikit-learn)
# Seleccionamos las columnas limpias que usará la IA + el Título original para mostrar al usuario
cols_exportar = ['title', 'rating_num', 'genres_clean', 'author_clean', 'description_clean', 'thumbnail']
df_maestro[cols_exportar].to_csv('dataset_entrenamiento_ia.csv', index=False)
print("Archivo 'dataset_entrenamiento_ia.csv' generado con éxito.")

# 3. Ejemplo de Filtrado
# print("\n--- Ejemplo de Filtrado ---")
# recomendaciones = filtrar_dataset(df_maestro, genero="fiction", palabra_clave="love")
# print(recomendaciones[['title', 'rating_num']].head())