import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('pastel')

# Lectura de datos // en el df tengo todos los datos del housing.csv
df = pd.read_csv('housing.csv')

# Identificación y manejo de datos faltantes
df.replace("?", np.nan, inplace=True)  # Reemplaza los "?" con NaN
missing_data = df.isnull()  # Identifica los valores nulos en el dataframe

# Encuentra las filas que no tienen valores en la columna 'total_bedrooms'
rows_without_total_bedrooms = df[df['total_bedrooms'].isnull()]


for column in missing_data.columns:
    n_missing = sum(missing_data[column])
    if n_missing > 0:
        print(f"{column}: {n_missing}/{missing_data.shape[0]} datos mancanti.")

# Imprime las filas que no tienen valores en la columna 'total_bedrooms'
print("Filas sin valores en 'total_bedrooms':")
print(rows_without_total_bedrooms)

# Gestión de Datos Faltantes: Eliminación y Sustitución
df_del_row = df.copy(deep=True)
df_del_row.dropna(axis=0, inplace=True)  # Elimina filas con datos faltantes

df_del_col = df.copy(deep=True)
df_del_col.dropna(axis=1, inplace=True)  # Elimina columnas con datos faltantes

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['total_bedrooms', 'housing_median_age']] = imputer.fit_transform(df[['total_bedrooms', 'housing_median_age']])

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[['ocean_proximity']] = imputer.fit_transform(df[['ocean_proximity']])

# Visualización de Outliers
sns.boxplot(data=df, y='total_bedrooms')
plt.savefig('boxplot_total_bedrooms.png')

sns.boxplot(data=df, y='median_income')
plt.savefig('boxplot_median_income.png')

sns.boxplot(data=df, y='median_income', x='ocean_proximity')
plt.savefig('boxplot_media_income_and_ocean_proximity.png')

sns.boxplot(data=df, y='total_bedrooms', x='ocean_proximity')
plt.savefig('boxplot_total_bedrooms_and_ocean_proximity.png')



# Codificación de Datos
label_encoder = LabelEncoder()
df_encoded_label = df.copy(deep=True)
df_encoded_label['ocean_proximity'] = label_encoder.fit_transform(df_encoded_label['ocean_proximity'])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
df_encoded_one = ct.fit_transform(df)
df_encoded_one = pd.DataFrame(df_encoded_one)

# Asignar automáticamente los nombres de las columnas
df_encoded_one.columns = list(df_encoded_one.columns)

# Muestra el DataFrame resultante después del preprocesamiento
print(df_encoded_one.head())
