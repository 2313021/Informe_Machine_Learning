# verificar_archivos.py
import os
import glob

print(" Buscando archivos disponibles...")
print("Directorio actual:", os.getcwd())
print("\nArchivos en el directorio:")
print("-" * 50)

# Listar todos los archivos
archivos = os.listdir()
for archivo in archivos:
    if archivo.endswith(('.json', '.parquet', '.csv', '.txt')):
        print(f" {archivo}")

# Buscar archivos que contengan "full" o "amazon"
print("\n Archivos que podrían ser el dataset:")
print("-" * 50)
patrones = ['*full*', '*amazon*', '*review*', '*data*']
for patron in patrones:
    for archivo in glob.glob(patron):
        if os.path.isfile(archivo):
            tamaño = os.path.getsize(archivo) / (1024 * 1024)  # Tamaño en MB
            print(f" {archivo} - {tamaño:.2f} MB")