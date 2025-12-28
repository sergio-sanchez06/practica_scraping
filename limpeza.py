import pandas as pd
import numpy as np
import re
import ast

def ejecutar_limpieza():
    print("🚀 Iniciando limpieza de datos para el RA4...")
    
    # 1. Cargar el archivo generado por el scraping
    try:
        df = pd.read_csv('books_google.csv')
    except FileNotFoundError:
        print("❌ Error: No se encuentra 'books_google.csv'. Ejecuta primero getData.py")
        return

    # 2. Eliminar duplicados (Fundamental en el RA4)
    antes = len(df)
    df = df.drop_duplicates(subset=['title', 'author'])
    print(f"✅ Se han eliminado {antes - len(df)} libros duplicados.")

    # 3. Limpiar la columna Rating (Pasar de texto a número)
    # Ejemplo: "4.35 avg rating" -> 4.35
    df['rating'] = df['rating'].str.extract(r'(\d+\.\d+)').astype(float)

    # 4. Limpiar los Géneros (Vienen de la API de Google como listas)
    # Ejemplo: "['Fiction']" -> "Fiction"
    def limpiar_lista_generos(valor):
        try:
            if pd.isna(valor) or valor == "[]": return "Unknown"
            # Convertimos el string de la lista a una lista real de Python
            lista = ast.literal_eval(valor)
            return lista[0] if len(lista) > 0 else "Unknown"
        except:
            return "Unknown"

    df['genres'] = df['genres'].apply(limpiar_lista_generos)

    # 5. Uso de NumPy para valores nulos (Requisito RA4)
    # Si la descripción está vacía, ponemos un texto por defecto usando np.nan
    df['description'] = df['description'].replace('', np.nan)
    df['description'] = df['description'].fillna("No description available")
    
    # Imputar ratings vacíos con la media usando NumPy
    media_rating = np.mean(df['rating'].dropna())
    df['rating'] = df['rating'].fillna(media_rating)

    # 6. Guardar el resultado final
    df.to_csv('books_cleaned.csv', index=False)
    print("💾 ¡Hecho! Archivo 'books_cleaned.csv' guardado y listo para clustering.")

if __name__ == "__main__":
    ejecutar_limpieza()