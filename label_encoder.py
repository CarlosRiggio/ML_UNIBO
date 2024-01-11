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

df = pd.read_csv('data_preproc.csv')

# %% Missing data: identificazione

df.replace("?", np.nan, inplace=True)  # sostituiamo il ?

missing_data = df.isnull()  # ricaviamo i nan nel dataframe

for column in missing_data.columns:
    n_missing = sum(missing_data[column])
    if n_missing > 0:
        print(f"{column}: {n_missing}/{missing_data.shape[0]} dati mancanti.")

# %% Missing data: gestione

# eliminazione dell'osservazione
df_del_row = df.copy(deep=True)
df_del_row.dropna(axis=0, inplace=True)
df_del_row.reset_index(inplace=True, drop=True)

# eliminazione della feature
df_del_col = df.copy(deep=True)
df_del_col.dropna(axis=1, inplace=True)

# sostituzione: dati numerici
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # median, constant, most_frequent
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# sostituzione: dati non numerici
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[['Country']] = imputer.fit_transform(df[['Country']])

# %% Outlier

sns.boxplot(data=df, y='Age')
plt.show()

sns.boxplot(data=df, y='Salary')
plt.show()

sns.boxplot(data=df, y='Salary', x='Country')
plt.show()

sns.boxplot(data=df, y='Age', x='Country')
plt.show()

# %% Data encoding

# label encoding
label_encoder = LabelEncoder()
df_encoded_label = df.copy(deep=True)
df_encoded_label['Country'] = label_encoder.fit_transform(df_encoded_label['Country'])

# one-hot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
df_encoded_one = ct.fit_transform(df)
df_encoded_one = pd.DataFrame(df_encoded_one)

df_encoded_one.columns = ['France', 'Germany', 'Spain', 'Age', 'Salary', 'Purchased']

# %%
