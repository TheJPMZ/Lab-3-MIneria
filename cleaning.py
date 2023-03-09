from quickda.clean_data import clean
from quickda.explore_numeric import eda_num
from quickda.explore_categoric import eda_cat
import pandas as pd
from pandas_profiling import ProfileReport

datos = pd.read_csv("wine_fraud.csv")

cat_cols = ["quality", "type"]
num_cols = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]

#ProfileReport(datos).to_file("report.html")

# Eliminamos las 992 filas duplicadas
datos = clean(datos, "duplicates")

# Estandarizamos las columnas
datos = clean(datos, "standardize")

# Graficos exploratorios
def exploration(dataframe: pd.DataFrame, categoric:list) -> None:
    
    eda_num(dataframe)

    for x in categoric:
        eda_cat(datos, x)
        
exploration(datos, cat_cols)

# Codificación de variables
def codify_bind(original_dataframe: pd.DataFrame, variables: list) -> pd.DataFrame:

    for variable in variables:

        dummies = pd.get_dummies(original_dataframe[[variable]])

        new_dataframe = pd.concat([original_dataframe, dummies], axis=1)

        new_dataframe = new_dataframe.drop([variable], axis=1)

        original_dataframe = new_dataframe

    return original_dataframe

datos = codify_bind(datos, cat_cols)

# Escalamiento de los datos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

edatos = pd.DataFrame(scaler.fit_transform(datos), columns=datos.columns)

edatos.describe()

datos.to_csv("clean_data.csv", index=False)
edatos.to_csv("scaled_data.csv", index=False)
