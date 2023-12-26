import csv

def encontrar_valores_faltantes(archivo_csv):
    filas_con_valores_faltantes = []

    with open(archivo_csv, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        encabezado = next(lector_csv)

        for numero_fila, fila in enumerate(lector_csv, start=2):  # Empezamos desde la segunda fila (índice 1)
            for indice, valor in enumerate(fila):
                if not valor:
                    filas_con_valores_faltantes.append((numero_fila, encabezado[indice]))

    return filas_con_valores_faltantes

archivo_csv = 'housing.csv'  # Reemplaza 'tu_archivo.csv' con la ruta correcta a tu archivo CSV

valores_faltantes = encontrar_valores_faltantes(archivo_csv)

if valores_faltantes:
    print("Filas con valores faltantes:")
    for fila, columna in valores_faltantes:
        print(f"Fila {fila}: Valor faltante en la columna '{columna}'")
else:
    print("No se encontraron valores faltantes en el archivo CSV.")
