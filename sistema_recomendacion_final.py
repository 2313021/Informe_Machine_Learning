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
        'full-00000-of-00001.parquet.as.json',
        'full*.json',
        '*.parquet.as.json', 
        'amazon*.json',
        'review*.json',
        '*.json'
    ]
    
    for patron in patrones:
        archivos = glob.glob(patron)
        for archivo in archivos:
            if os.path.isfile(archivo) and os.path.getsize(archivo) > 0:
                print(f"✅ Archivo encontrado: {archivo}")
                return archivo
    
    print("❌ No se encontró el archivo del dataset")
    return None

def cargar_dataset(ruta_archivo):
    """Cargar el dataset con manejo de errores"""
    try:
        print(f"📥 Cargando dataset desde: {ruta_archivo}")
        
        # Intentar diferentes métodos de carga
        if ruta_archivo.endswith('.json'):
            df = pd.read_json(ruta_archivo, lines=True)
        elif ruta_archivo.endswith('.parquet'):
            df = pd.read_parquet(ruta_archivo)
        elif ruta_archivo.endswith('.csv'):
            df = pd.read_csv(ruta_archivo)
        else:
            # Intentar leer como JSON por defecto
            df = pd.read_json(ruta_archivo, lines=True)
            
        print(f"✅ Dataset cargado: {len(df)} registros")
        return df
        
    except Exception as e:
        print(f"❌ Error cargando el archivo: {e}")
        return None

def crear_dataset_ejemplo():
    """Crear dataset de ejemplo si no se encuentra el archivo real"""
    print("🔄 Creando dataset de ejemplo...")
    
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
    print(f"✅ Dataset de ejemplo creado: {len(df)} productos")
    return df

# --- PROGRAMA PRINCIPAL ---
print("🚀 SISTEMA DE RECOMENDACIÓN APRIORI")
print("=" * 60)

# 1. Buscar y cargar dataset
ruta_dataset = encontrar_archivo_dataset()

if ruta_dataset:
    df = cargar_dataset(ruta_dataset)
else:
    print("📦 Usando dataset de ejemplo...")
    df = crear_dataset_ejemplo()

if df is None:
    print("❌ No se pudo cargar ningún dataset. Saliendo...")
    exit()

# 2. Mostrar información del dataset
print(f"\n📊 INFORMACIÓN DEL DATASET:")
print(f"   - Total de registros: {len(df)}")
print(f"   - Columnas disponibles: {list(df.columns)}")
print(f"   - Primeras filas:")
print(df.head(3))

# 3. Simular transacciones de usuarios
print(f"\n🔄 SIMULANDO TRANSACCIONES DE USUARIOS...")

def simular_transacciones_inteligentes(df, n_usuarios=300):
    """Simular transacciones realistas basadas en categorías y ratings"""
    
    transacciones = []
    productos_disponibles = df['title'].tolist()
    
    print(f"   🛍️  Productos disponibles: {len(productos_disponibles)}")
    
    # Crear patrones de compra por categoría
    categorias = df['main_category'].unique()
    patrones_compra = {}
    
    for categoria in categorias:
        productos_categoria = df[df['main_category'] == categoria]['title'].tolist()
        if len(productos_categoria) >= 2:
            patrones_compra[categoria] = productos_categoria
    
    # Generar transacciones para cada usuario
    for i in range(n_usuarios):
        user_id = f"U{i+1:04d}"
        
        # Cada usuario compra 3-6 productos
        n_compras = np.random.randint(3, 7)
        productos_comprados = []
        
        # Patrón 1: Comprar productos de la misma categoría
        if patrones_compra:
            categoria_elegida = np.random.choice(list(patrones_compra.keys()))
            productos_categoria = patrones_compra[categoria_elegida]
            n_en_categoria = min(2, len(productos_categoria))
            
            if n_en_categoria > 0:
                productos_cat = np.random.choice(productos_categoria, n_en_categoria, replace=False)
                productos_comprados.extend(productos_cat)
        
        # Patrón 2: Productos con alto rating
        productos_alto_rating = df[df['average_rating'] >= 4.0]['title'].tolist()
        if productos_alto_rating:
            n_alto_rating = min(2, len(productos_alto_rating))
            productos_rating = np.random.choice(productos_alto_rating, n_alto_rating, replace=False)
            productos_comprados.extend(productos_rating)
        
        # Patrón 3: Productos populares (muchos ratings)
        if 'rating_number' in df.columns:
            productos_populares = df.nlargest(10, 'rating_number')['title'].tolist()
            if productos_populares:
                productos_comprados.extend(np.random.choice(productos_populares, 1, replace=False))
        
        # Eliminar duplicados y asegurar número correcto
        productos_comprados = list(set(productos_comprados))
        if len(productos_comprados) > n_compras:
            productos_comprados = np.random.choice(productos_comprados, n_compras, replace=False)
        
        # Agregar a transacciones
        for producto in productos_comprados:
            categoria = df[df['title'] == producto]['main_category'].iloc[0]
            rating = df[df['title'] == producto]['average_rating'].iloc[0]
            
            transacciones.append({
                'user_id': user_id,
                'product_title': producto,
                'main_category': categoria,
                'rating': rating
            })
    
    df_transacciones = pd.DataFrame(transacciones)
    print(f"   ✅ Transacciones simuladas: {len(df_transacciones)}")
    print(f"   👥 Usuarios únicos: {df_transacciones['user_id'].nunique()}")
    print(f"   🛍️  Productos únicos: {df_transacciones['product_title'].nunique()}")
    
    return df_transacciones

# Generar transacciones
df_transacciones = simular_transacciones_inteligentes(df, n_usuarios=200)

# 4. Preparar datos para Apriori
print(f"\n🔧 PREPARANDO DATOS PARA ALGORITMO APRIORI...")

# Agrupar por usuario
transacciones_agrupadas = df_transacciones.groupby('user_id')['product_title'].apply(list).tolist()

print(f"   📊 Total de transacciones: {len(transacciones_agrupadas)}")
print(f"   🎯 Ejemplo: Usuario {df_transacciones['user_id'].iloc[0]} compró {len(transacciones_agrupadas[0])} productos")

# Codificar transacciones
te = TransactionEncoder()
te_array = te.fit(transacciones_agrupadas).transform(transacciones_agrupadas)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print(f"   🔤 Productos únicos en transacciones: {len(te.columns_)}")

# 5. Ejecutar algoritmo Apriori
print(f"\n🎯 EJECUTANDO ALGORITMO APRIORI...")

# Parámetros ajustables
min_support = 0.03
min_confidence = 0.4

print(f"   ⚙️  Parámetros: min_support={min_support}, min_confidence={min_confidence}")

frequent_itemsets = apriori(
    df_encoded, 
    min_support=min_support,
    use_colnames=True,
    max_len=2
)

print(f"   ✅ Itemsets frecuentes encontrados: {len(frequent_itemsets)}")

if len(frequent_itemsets) > 0:
    # Generar reglas
    rules = association_rules(
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=min_confidence
    )
    
    # Filtrar reglas útiles
    rules = rules[rules['lift'] > 1.0]
    
    print(f"   🔗 Reglas de asociación generadas: {len(rules)}")
    
    # 6. Mostrar resultados
    if len(rules) > 0:
        print(f"\n📈 MEJORES REGLAS DE ASOCIACIÓN:")
        print("=" * 70)
        
        rules_sorted = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        
        for i, (idx, rule) in enumerate(rules_sorted.head(8).iterrows(), 1):
            antecedente = list(rule['antecedents'])[0]
            consecuente = list(rule['consequents'])[0]
            
            print(f"{i}. SI compras: {antecedente}")
            print(f"   ENTONCES probablemente compres: {consecuente}")
            print(f"   📊 Soporte: {rule['support']:.3f} | 🎯 Confianza: {rule['confidence']:.3f} | 🚀 Lift: {rule['lift']:.3f}")
            print()
            
    else:
        print("❌ No se generaron reglas con los parámetros actuales")
        print("💡 Sugerencia: Reduce min_confidence a 0.3 o min_support a 0.02")
        
else:
    print("❌ No se encontraron itemsets frecuentes")
    print("💡 Sugerencia: Reduce min_support a 0.02")

# 7. Sistema de recomendación
def recomendar_productos(producto_entrada, rules, df_original, top_n=3):
    """Sistema de recomendación simple"""
    print(f"\n🎯 RECOMENDACIONES PARA: {producto_entrada}")
    print("-" * 50)
    
    recomendaciones = []
    
    for idx, rule in rules.iterrows():
        antecedentes = list(rule['antecedents'])
        
        if producto_entrada in antecedentes:
            for consecuente in rule['consequents']:
                if consecuente != producto_entrada:
                    # Buscar info del producto
                    producto_info = df_original[df_original['title'] == consecuente]
                    if not producto_info.empty:
                        info = producto_info.iloc[0]
                        recomendaciones.append({
                            'producto': consecuente,
                            'categoria': info['main_category'],
                            'rating': info['average_rating'],
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
            print(f"   📁 Categoría: {rec['categoria']} | ⭐ Rating: {rec['rating']}")
            print(f"   🎯 Confianza: {rec['confidence']:.3f} | 🚀 Lift: {rec['lift']:.3f}")
            print()
    else:
        print("❌ No se encontraron recomendaciones para este producto")
        print("💡 Prueba con otro producto o ajusta los parámetros del algoritmo")
    
    return recomendaciones_unicas[:top_n]

# 8. Probar el sistema
if 'rules' in locals() and len(rules) > 0:
    print(f"\n🧪 PROBANDO SISTEMA DE RECOMENDACIÓN...")
    
    # Usar productos populares para prueba
    productos_populares = df_transacciones['product_title'].value_counts().head(3).index.tolist()
    
    for producto in productos_populares:
        recomendar_productos(producto, rules, df, top_n=2)

# 9. Guardar resultados
print(f"\n💾 GUARDANDO RESULTADOS...")

df_transacciones.to_csv('transacciones_.csv', index=False)
print("✅ Transacciones guardadas en 'transacciones_.csv'")

if 'rules' in locals() and len(rules) > 0:
    rules.to_csv('reglas_asociacion.csv', index=False)
    print("✅ Reglas de asociación guardadas en 'reglas_asociacion.csv'")

print(f"\n🎉 SISTEMA DE RECOMENDACIÓN COMPLETADO!")
print(f"   📈 Itemsets frecuentes: {len(frequent_itemsets)}")
print(f"   🔗 Reglas generadas: {len(rules) if 'rules' in locals() else 0}")