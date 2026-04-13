import numpy as np
from typing import Tuple

def generate_synthetic_dataset(dataset_size: int, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un conjunto de datos sintético para tareas de regresión.
    La variable objetivo (y) se crea a partir de las primeras 3 características
    sumándole ruido gaussiano para que el modelo tenga un patrón real que encontrar.

    Args:
        dataset_size (int): Cantidad de muestras a generar.
        input_dim (int): Número de dimensiones/características por muestra.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) donde X son las características y y es la variable a predecir.
    """
    # X son números aleatorios entre 0 y 1 uniformes
    X = np.random.rand(dataset_size, input_dim).astype(np.float32)
    
    # y = sumatoria de características 0,1,2 * 2 + ruido aleatorio con distribución normal
    y = (np.sum(X[:, :3], axis=1, keepdims=True) * 2.0 + 
         np.random.normal(0, 0.1, (dataset_size, 1))).astype(np.float32)
         
    return X, y
