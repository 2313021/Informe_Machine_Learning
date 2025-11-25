# sistema_recomendacion_final.py
import pandas as pd
import numpy as np
import os
import glob
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

def encontrar_archivo_dataset():
    """Buscar automáticamente el archivo del dataset"""
    print(" Buscando archivo del dataset...")
    
    patrones = [
        'full-00000-of-00001.parquet',
        'full*.parquet',
        '*.parquet', 
        'amazon*.parquet',
        'review*.parquet',
        '*.json',
        '*.csv'
    ]
    
    for patron in patrones:
        archivos = glob.glob(patron)
        for archivo in archivos:
            if os.path.isfile(archivo) and os.path.getsize(archivo) > 0:
                print(f" Archivo encontrado: {archivo}")
                return archivo
    
    print(" No se encontró el archivo del dataset")
    return None

def cargar_dataset(ruta_archivo):
    """Cargar el dataset con manejo de errores"""
    try:
        print(f" Cargando dataset desde: {ruta_archivo}")
        
        # Intentar diferentes métodos de carga
        if ruta_archivo.endswith('.parquet'):
            df = pd.read_parquet(ruta_archivo)
        elif ruta_archivo.endswith('.json'):
            df = pd.read_json(ruta_archivo, lines=True)
        elif ruta_archivo.endswith('.csv'):
            df = pd.read_csv(ruta_archivo)
        else:
            # Intentar leer como Parquet por defecto
            df = pd.read_parquet(ruta_archivo)
            
        print(f" Dataset cargado: {len(df)} registros")
        return df
        
    except Exception as e:
        print(f" Error cargando el archivo: {e}")
        return None

def crear_dataset_ejemplo():
    """Crear dataset de ejemplo si no se encuentra el archivo real"""
    print(" Creando dataset de ejemplo...")
    
    # Datos de ejemplo realistas de Amazon
    datos_ejemplo = {
        'main_category': ['Electronics', 'Electronics', 'Books', 'Books', 'Home', 'Electronics', 'Books', 'Home'],
        'title': [
            'Wireless Bluetooth Headphones',
            'Smartphone Case', 
            'Python Programming Book',
            'Machine Learning Guide',
            'Coffee Maker',
            'USB-C Charging Cable',
            'Data Science Handbook',
            'Desk Lamp'
        ],
        'average_rating': [4.5, 4.2, 4.7, 4.4, 4.3, 4.1, 4.6, 4.0],
        'rating_number': [1250, 890, 450, 320, 670, 1200, 280, 540],
        'price': ['$29.99', '$15.99', '$45.99', '$39.99', '$49.99', '$12.99', '$54.99', '$24.99']
    }
    
    df = pd.DataFrame(datos_ejemplo)
    print(f" Dataset de ejemplo creado: {len(df)} productos")
    return df

def procesar_dataset_real(df):
    """Procesar el dataset real para extraer transacciones reales"""
    print(f"\n PROCESANDO DATASET REAL...")
    
    # Verificar columnas disponibles
    print(f" Columnas disponibles: {list(df.columns)}")
    
    # Buscar columnas que puedan contener información de transacciones
    columnas_posibles = {
        'user_id': ['user_id', 'userID', 'userId', 'customer_id', 'reviewerID'],
        'product_id': ['product_id', 'productID', 'productId', 'asin', 'item_id'],
        'product_title': ['title', 'product_title', 'product_name', 'name'],
        'category': ['main_category', 'category', 'categories', 'mainCategory'],
        'rating': ['rating', 'average_rating', 'overall', 'reviewRating']
    }
    
    # Mapear columnas reales
    columnas_mapeadas = {}
    for tipo, opciones in columnas_posibles.items():
        for opcion in opciones:
            if opcion in df.columns:
                columnas_mapeadas[tipo] = opcion
                break
    
    print(f" Columnas mapeadas: {columnas_mapeadas}")
    
    # Verificar si tenemos suficiente información para transacciones
    if 'user_id' not in columnas_mapeadas or 'product_id' not in columnas_mapeadas:
        print(" AVISO: No se encontraron columnas de usuario y producto para transacciones reales")
        print(" Se usarán datos simulados basados en el dataset real")
        return simular_transacciones_inteligentes(df)
    
    # Crear transacciones reales agrupando por usuario
    print(f" Creando transacciones reales...")
    
    # Agrupar productos por usuario
    columna_usuario = columnas_mapeadas['user_id']
    columna_producto = columnas_mapeadas.get('product_title', columnas_mapeadas['product_id'])
    
    transacciones_reales = df.groupby(columna_usuario)[columna_producto].apply(list).reset_index()
    
    print(f" Transacciones reales encontradas: {len(transacciones_reales)}")
    print(f" Ejemplo de transacción: {transacciones_reales.iloc[0][columna_producto][:3]}...")
    
    # Crear DataFrame de transacciones para el análisis
    transacciones_lista = []
    for _, row in transacciones_reales.iterrows():
        user_id = row[columna_usuario]
        productos = row[columna_producto]
        
        for producto in productos:
            # Buscar información adicional del producto
            producto_info = df[df[columna_producto] == producto].iloc[0]
            
            transaccion = {
                'user_id': user_id,
                'product_title': producto
            }
            
            # Agregar información adicional si está disponible
            if 'category' in columnas_mapeadas:
                transaccion['main_category'] = producto_info[columnas_mapeadas['category']]
            if 'rating' in columnas_mapeadas:
                transaccion['rating'] = producto_info[columnas_mapeadas['rating']]
            
            transacciones_lista.append(transaccion)
    
    df_transacciones = pd.DataFrame(transacciones_lista)
    
    print(f" Registros de transacciones: {len(df_transacciones)}")
    print(f" Usuarios únicos: {df_transacciones['user_id'].nunique()}")
    print(f" Productos únicos: {df_transacciones['product_title'].nunique()}")
    
    return df_transacciones

def simular_transacciones_inteligentes(df):
    """Función de respaldo si no hay suficientes datos reales"""
    print(" Simulando transacciones basadas en datos reales...")
    
    transacciones = []
    productos_disponibles = df['title'].tolist() if 'title' in df.columns else df.iloc[:, 0].tolist()
    
    print(f" Productos disponibles: {len(productos_disponibles)}")
    
    # Generar transacciones para 200 usuarios
    for i in range(200):
        user_id = f"U{i+1:04d}"
        n_compras = np.random.randint(2, 6)
        
        productos_comprados = np.random.choice(productos_disponibles, n_compras, replace=False)
        
        for producto in productos_comprados:
            transacciones.append({
                'user_id': user_id,
                'product_title': producto
            })
    
    df_transacciones = pd.DataFrame(transacciones)
    print(f" Transacciones simuladas: {len(df_transacciones)}")
    
    return df_transacciones

# --- PROGRAMA PRINCIPAL ---
print(" SISTEMA DE RECOMENDACIÓN APRIORI - DATOS REALES")
print("=" * 60)

# 1. Buscar y cargar dataset
ruta_dataset = encontrar_archivo_dataset()

if ruta_dataset:
    df = cargar_dataset(ruta_dataset)
else:
    print(" Usando dataset de ejemplo...")
    df = crear_dataset_ejemplo()

if df is None:
    print(" No se pudo cargar ningún dataset. Saliendo...")
    exit()

# 2. Mostrar información del dataset
print(f"\n INFORMACIÓN DEL DATASET:")
print(f"   - Total de registros: {len(df)}")
print(f"   - Columnas disponibles: {list(df.columns)}")
print(f"   - Primeras filas:")
print(df.head(3))

# 3. Procesar transacciones reales
df_transacciones = procesar_dataset_real(df)

# 4. Preparar datos para Apriori
print(f"\n PREPARANDO DATOS PARA ALGORITMO APRIORI...")

# Agrupar por usuario
transacciones_agrupadas = df_transacciones.groupby('user_id')['product_title'].apply(list).tolist()

print(f"    Total de transacciones: {len(transacciones_agrupadas)}")
print(f"    Ejemplo: Usuario {df_transacciones['user_id'].iloc[0]} compró {len(transacciones_agrupadas[0])} productos")

# Codificar transacciones
te = TransactionEncoder()
te_array = te.fit(transacciones_agrupadas).transform(transacciones_agrupadas)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print(f"    Productos únicos en transacciones: {len(te.columns_)}")

# 5. Ejecutar algoritmo Apriori
print(f"\n EJECUTANDO ALGORITMO APRIORI...")

# Parámetros ajustables
min_support = 0.03
min_confidence = 0.4

print(f"     Parámetros: min_support={min_support}, min_confidence={min_confidence}")

frequent_itemsets = apriori(
    df_encoded, 
    min_support=min_support,
    use_colnames=True,
    max_len=2
)

print(f"    Itemsets frecuentes encontrados: {len(frequent_itemsets)}")

if len(frequent_itemsets) > 0:
    # Generar reglas
    rules = association_rules(
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=min_confidence
    )
    
    # Filtrar reglas útiles
    rules = rules[rules['lift'] > 1.0]
    
    print(f"    Reglas de asociación generadas: {len(rules)}")
    
    # 6. Mostrar resultados
    if len(rules) > 0:
        print(f"\n MEJORES REGLAS DE ASOCIACIÓN:")
        print("=" * 70)
        
        rules_sorted = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        
        for i, (idx, rule) in enumerate(rules_sorted.head(8).iterrows(), 1):
            antecedente = list(rule['antecedents'])[0]
            consecuente = list(rule['consequents'])[0]
            
            print(f"{i}. SI compras: {antecedente}")
            print(f"   ENTONCES probablemente compres: {consecuente}")
            print(f"    Soporte: {rule['support']:.3f} |  Confianza: {rule['confidence']:.3f} | 🚀 Lift: {rule['lift']:.3f}")
            print()
            
    else:
        print(" No se generaron reglas con los parámetros actuales")
        print(" Sugerencia: Reduce min_confidence a 0.3 o min_support a 0.02")
        
else:
    print(" No se encontraron itemsets frecuentes")
    print(" Sugerencia: Reduce min_support a 0.02")

# 7. Sistema de recomendación
def recomendar_productos(producto_entrada, rules, df_original, top_n=3):
    """Sistema de recomendación simple"""
    print(f"\n RECOMENDACIONES PARA: {producto_entrada}")
    print("-" * 50)
    
    recomendaciones = []
    
    for idx, rule in rules.iterrows():
        antecedentes = list(rule['antecedents'])
        
        if producto_entrada in antecedentes:
            for consecuente in rule['consequents']:
                if consecuente != producto_entrada:
                    # Buscar info del producto
                    producto_info = df_original[df_original['title'] == consecuente] if 'title' in df_original.columns else df_original[df_original.iloc[:, 0] == consecuente]
                    if not producto_info.empty:
                        info = producto_info.iloc[0]
                        recomendaciones.append({
                            'producto': consecuente,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'score': rule['confidence'] * rule['lift']
                        })
    
    # Eliminar duplicados y ordenar
    recomendaciones_unicas = []
    productos_vistos = set()
    
    for rec in recomendaciones:
        if rec['producto'] not in productos_vistos:
            productos_vistos.add(rec['producto'])
            recomendaciones_unicas.append(rec)
    
    recomendaciones_unicas.sort(key=lambda x: x['score'], reverse=True)
    
    # Mostrar resultados
    if recomendaciones_unicas:
        for i, rec in enumerate(recomendaciones_unicas[:top_n], 1):
            print(f"{i}. {rec['producto']}")
            print(f"    Confianza: {rec['confidence']:.3f} |  Lift: {rec['lift']:.3f}")
            print()
    else:
        print(" No se encontraron recomendaciones para este producto")
        print(" Sugerencia: Prueba con otro producto o ajusta los parámetros del algoritmo")
    
    return recomendaciones_unicas[:top_n]

# 8. Probar el sistema
if 'rules' in locals() and len(rules) > 0:
    print(f"\n PROBANDO SISTEMA DE RECOMENDACIÓN...")
    
    # Usar productos populares para prueba
    productos_populares = df_transacciones['product_title'].value_counts().head(3).index.tolist()
    
    for producto in productos_populares:
        recomendar_productos(producto, rules, df, top_n=2)

# 9. Guardar resultados
print(f"\n GUARDANDO RESULTADOS...")

df_transacciones.to_csv('transacciones_reales.csv', index=False)
print(" Transacciones guardadas en 'transacciones_reales.csv'")

if 'rules' in locals() and len(rules) > 0:
    rules.to_csv('reglas_asociacion_reales.csv', index=False)
    print(" Reglas de asociación guardadas en 'reglas_asociacion_reales.csv'")

print(f"\n SISTEMA DE RECOMENDACIÓN COMPLETADO!")
print(f"   Itemsets frecuentes: {len(frequent_itemsets)}")
print(f"   Reglas generadas: {len(rules) if 'rules' in locals() else 0}")