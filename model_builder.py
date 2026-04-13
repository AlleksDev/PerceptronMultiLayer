import tensorflow as tf
from typing import List

class ModelBuilder:
    """Clase encargada de construir y compilar la arquitectura Sequential de la red neuronal.
    
    Attributes:
        input_dim (int): Número de características de entrada.
        output_dim (int): Número de neuronas de salida.
        layers_config (List[int]): Lista con el número de neuronas por capa oculta.
        activations (List[str]): Lista con las funciones de activación por capa oculta.
    """
    
    def __init__(self, input_dim: int, output_dim: int, layers_config: List[int], activations: List[str]):
        """Inicializa los parámetros de construcción del modelo."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_config = layers_config
        self.activations = activations

    def build(self) -> tf.keras.Model:
        """Construye y compila el modelo Sequential.
        
        Returns:
            tf.keras.Model: Modelo de Keras compilado para regresión con MSE.
            
        Raises:
            ValueError: Si la longitud de las capas y activaciones no coincide.
        """
        if len(self.layers_config) != len(self.activations):
            raise ValueError("La lista de neuronas y activaciones debe tener el mismo tamaño.")

        model = tf.keras.Sequential()
        
        # Capa de entrada
        model.add(tf.keras.layers.InputLayer(input_shape=(self.input_dim,)))
        
        # Capas ocultas dinámicas
        for units, activation in zip(self.layers_config, self.activations):
            model.add(tf.keras.layers.Dense(units=units, activation=activation))
            
        # Capa de salida (Regresión -> activación lineal)
        model.add(tf.keras.layers.Dense(units=self.output_dim, activation='linear'))
        
        # Compilación: Optimizador Adam y métricas MAE complementarias
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
