import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging (warnings e info)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from typing import List, Dict, Any

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


class CrossValidator:
    """Clase para gestionar la validación cruzada del modelo utilizando K-Fold.
    
    Attributes:
        builder (ModelBuilder): Instancia de constructor para generar modelos "vírgenes" por fold.
        n_splits (int): Total de bloques para la validación cruzada.
    """
    
    def __init__(self, builder: ModelBuilder, n_splits: int):
        """Inicializa el validador cruzado inyectando la dependencia del builder."""
        self.builder = builder
        self.n_splits = n_splits

    def evaluate(self, X: np.ndarray, y: np.ndarray, epochs: int = 15, batch_size: int = 32) -> Dict[str, float]:
        """Realiza K-Fold Cross Validation iterando sobre cada split.
        
        Args:
            X (np.ndarray): Datos de entrada (características).
            y (np.ndarray): Datos objetivo (etiquetas/salida).
            epochs (int): Cantidad de épocas a entrenar obligatorias por fold.
            batch_size (int): Tamaño del lote para entrenamiento.
            
        Returns:
            Dict[str, float]: Diccionario que contiene las métricas promediadas (loss y mae).
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_losses = []
        fold_maes = []
        
        print(f"\n--- Iniciando Validación Cruzada ({self.n_splits} Folds) ---")
        
        # Iteración explícita sobre splits
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            print(f"\n[Fold {fold}/{self.n_splits}]")
            
            # Segmentar datos
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Recrear el modelo para limpiar pesos, evitando filtrado de conocimiento previo
            model = self.builder.build()
            
            print(f"  -> Entrenando: {len(train_idx)} muestras | Validando: {len(val_idx)} muestras")
            
            # Entrenamiento silencioso (verbose=0) para no saturar la salida
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Evaluación final de val para este fold
            results = model.evaluate(X_val, y_val, verbose=0)
            
            loss_val = results[0] if isinstance(results, list) else results
            mae_val = results[1] if isinstance(results, list) and len(results) > 1 else 0.0
            
            print(f"  -> MSE (Loss): {loss_val:.4f}  |  MAE: {mae_val:.4f}")
            
            fold_losses.append(loss_val)
            fold_maes.append(mae_val)
            
        # Calcular los promedios globales del sistema tras la evaluación completa
        avg_loss = float(np.mean(fold_losses))
        avg_mae = float(np.mean(fold_maes))
        
        print(f"\n--- Resumen Final del K-Fold ({self.n_splits} splits) ---")
        print(f"Promedio Loss (MSE): {avg_loss:.4f}  (+/- {np.std(fold_losses):.4f})")
        print(f"Promedio MAE:        {avg_mae:.4f}  (+/- {np.std(fold_maes):.4f})")
        
        return {
            "mean_loss": avg_loss,
            "mean_mae": avg_mae
        }


if __name__ == "__main__":
    # ---------------------------------------------
    # 1. Parámetros Construcción (Arquitectura)
    # ---------------------------------------------
    input_dim: int = 10
    output_dim: int = 1
    layers_config: List[int] = [64, 32]
    activations: List[str] = ['relu', 'relu']
    dataset_size: int = 1000
    
    # ---------------------------------------------
    # 2. Configuración de Validación Cruzada
    # ---------------------------------------------
    n_splits: int = 5
    train_folds: int = 4
    val_folds: int = 1
    
    # Lógica que garantiza la proporción de bloques respetada para K-Fold.
    # K-Fold divide el dataset en n_splits, utilizando (n_splits - val_folds) para el entrenamiento.
    assert train_folds + val_folds == n_splits, "La proporción de bloques (train y val) debe sumar exactamente el total de 'n_splits'."
    
    print("========================================")
    print(" CONFIGURACIÓN DEL SISTEMA DE APRENDIZAJE")
    print("========================================")
    print(f"* Arquitectura: {layers_config} neuronas ({activations})")
    print(f"* K-Fold config: {n_splits} bloques totales.")
    print(f"* Repartición: {train_folds} bloque(s) de entrenamiento y {val_folds} de validación por iteración.")
    print("========================================\n")
    
    # ---------------------------------------------
    # 3. Generación de Datos Sintéticos para Regresión
    # ---------------------------------------------
    X_synthetic = np.random.rand(dataset_size, input_dim).astype(np.float32)
    # Variable objetivo sintética para que el modelo tenga un patrón real que capturar y la loss baje.
    y_synthetic = (np.sum(X_synthetic[:, :3], axis=1, keepdims=True) * 2.0 + 
                   np.random.normal(0, 0.1, (dataset_size, 1))).astype(np.float32)

    # ---------------------------------------------
    # 4. Inyección de Dependencias
    # ---------------------------------------------
    # Builder: Desacopla la lógica algorítmica de la lógica del flujo de datos
    builder = ModelBuilder(
        input_dim=input_dim,
        output_dim=output_dim,
        layers_config=layers_config,
        activations=activations
    )
    
    print("[Visualización de la Arquitectura Inicializando un modelo base]")
    dummy_model = builder.build()
    dummy_model.summary()

    # ---------------------------------------------
    # 5. Ejecución Central y Validación
    # ---------------------------------------------
    # CrossValidator: Recibe las responsabilidades de flujo
    validator = CrossValidator(builder=builder, n_splits=n_splits)
    
    final_metrics = validator.evaluate(
        X=X_synthetic, 
        y=y_synthetic, 
        epochs=15, 
        batch_size=32
    )
