import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging (warnings e info)

from typing import List
import matplotlib.pyplot as plt
from src.model_builder import ModelBuilder
from src.cross_validator import CrossValidator
from data.data_generator import load_dataset

if __name__ == "__main__":
    # ---------------------------------------------
    # 1. Parámetros Construcción (Arquitectura)
    # ---------------------------------------------
    input_dim: int = 10
    output_dim: int = 3 # 3 clases: Store, Online, Hybrid
    layers_config: List[int] = [16]
    activations: List[str] = ['relu']
    
    # Parámetros adicionales para clasificación
    output_activation: str = 'softmax'
    loss_function: str = 'sparse_categorical_crossentropy'
    metrics: List[str] = ['accuracy']
    
    dataset_size: int = 1000
    
    # ---------------------------------------------
    # 2. Configuración de Validación Cruzada
    # ---------------------------------------------
    n_splits: int = 5
    train_folds: int = 4
    val_folds: int = 1
    
    # Lógica que garantiza la proporción de bloques respetada para K-Fold.
    assert train_folds + val_folds == n_splits, "La proporción de bloques (train y val) debe sumar exactamente el total de 'n_splits'."
    
    print("========================================")
    print(" CONFIGURACIÓN DEL SISTEMA DE APRENDIZAJE")
    print("========================================")
    print(f"* Arquitectura: {layers_config} neuronas ({activations})")
    print(f"* K-Fold config: {n_splits} bloques totales.")
    print(f"* Repartición: {train_folds} bloque(s) de entrenamiento y {val_folds} de validación por iteración.")
    print("========================================\n")
    
    # ---------------------------------------------
    # 3. Carga del Dataset Real
    # ---------------------------------------------
    X_real, y_real = load_dataset()
    
    # Reducimos las características a solo 6 para la capa de entrada
    X_real = X_real[:, :6]
    input_dim = X_real.shape[1]
    
    # ---------------------------------------------
    # 4. Inyección de Dependencias
    # ---------------------------------------------
    # Builder: Desacopla la lógica algorítmica de la lógica del flujo de datos
    builder = ModelBuilder(
        input_dim=input_dim,
        output_dim=output_dim,
        layers_config=layers_config,
        activations=activations,
        output_activation=output_activation,
        loss=loss_function,
        metrics=metrics
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
        X=X_real, 
        y=y_real, 
        epochs=15, 
        batch_size=32
    )

    # ---------------------------------------------
    # 6. Visualización de la evolución del error
    # ---------------------------------------------
    best_hist = final_metrics.get("best_history")
    if best_hist:
        plt.figure(figsize=(10, 6))
        plt.plot(best_hist['loss'], label='Loss Entrenamiento')
        if 'val_loss' in best_hist:
            plt.plot(best_hist['val_loss'], label='Loss Validación')
        plt.title('Evolución del Error - Mejor Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (Sparse Categorical Crossentropy)')
        plt.legend()
        plt.grid(True)
        plt.show()

