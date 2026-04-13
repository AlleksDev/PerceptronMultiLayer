import numpy as np
import pandas as pd
import os
from typing import Tuple

def load_dataset(csv_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el dataset desde el archivo CSV y convierte los valores categóricos a numéricos.

    Args:
        csv_path (str, optional): Ruta al archivo CSV. Por defecto asume que está en el mismo directorio.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) donde X son las características y y es la variable a predecir.
    """
    if csv_path is None:
        # Por defecto, busca en el mismo directorio donde está este script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'Consumer_Shopping_Trends_2026.csv')

    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Mapeos para transformar valores de texto a numéricos
    gender_map = {'Male': 1, 'Female': 2, 'Other': 3}
    city_tier_map = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}
    shopping_pref_map = {'Store': 1, 'Online': 2, 'Hybrid': 3}

    # Aplicar los mapeos
    df['gender'] = df['gender'].map(gender_map)
    df['city_tier'] = df['city_tier'].map(city_tier_map)
    df['shopping_preference'] = df['shopping_preference'].map(shopping_pref_map)

    # Llenar posibles datos faltantes con 0 (o el valor que prefieras)
    df.fillna(0, inplace=True)

    # Separar en características (X) y la variable objetivo (y)
    # Suponiendo que queremos predecir la última columna ('shopping_preference')
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1:].values.astype(np.float32)
         
    return X, y

