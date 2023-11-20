import pandas as pd

# Reemplaza 'tu_archivo.csv' con la ruta de tu archivo CSV
archivo_csv = 'MOVIE_TRAIN_TEST.csv'

# Cargar el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Especificar el número mínimo de instancias que deseas para cada clase
min_instancias = 4624

# Inicializar un DataFrame vacío para almacenar los resultados
df_balanceado = pd.DataFrame()

# Iterar sobre las clases
for clase in df['class_index'].unique():
    # Filtrar el DataFrame para obtener solo las instancias de la clase actual
    df_clase = df[df['class_index'] == clase]
    
    # Verificar si la clase actual tiene más instancias que el mínimo deseado
    if len(df_clase) > min_instancias:
        # Submuestrear aleatoriamente el DataFrame de la clase actual
        df_muestreo_clase = df_clase.sample(min_instancias, random_state=42)
    else:
        # Si la clase tiene menos instancias que el mínimo, tomar todas las instancias
        df_muestreo_clase = df_clase
    
    # Concatenar el DataFrame de muestreo de la clase actual al DataFrame balanceado
    df_balanceado = pd.concat([df_balanceado, df_muestreo_clase])

# Reorganizar el DataFrame resultante
df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

df_balanceado.to_csv('MOVIE_BALANCE.csv',index=False)

# Imprimir el DataFrame resultante
print(df_balanceado.head())

# Verificar la frecuencia de clases en el DataFrame balanceado
print(df_balanceado['class_index'].value_counts())
