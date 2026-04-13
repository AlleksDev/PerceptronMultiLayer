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
    
    def __init__(self, input_dim: int, output_dim: int, layers_config: List[int], activations: List[str], output_activation: str = 'linear', loss: str = 'mse', metrics: List[str] = None):
        """Inicializa los parámetros de construcción del modelo."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_config = layers_config
        self.activations = activations
        self.output_activation = output_activation
        self.loss = loss
        self.metrics = metrics if metrics is not None else ['mae']

    def build(self) -> tf.keras.Model:
        """Construye y compila el modelo Sequential.
        
        Returns:
            tf.keras.Model: Modelo de Keras compilado.
            
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
            
        # Capa de salida configurable (Regresión/Clasificación según `output_activation`)
        model.add(tf.keras.layers.Dense(units=self.output_dim, activation=self.output_activation))
        
        # Compilación dinámica: Optimizador Adam, config de loss y metrics según los atributos instanciados
        model.compile(optimizer='adam', loss=self.loss, metrics=self.metrics)
        
        return model
