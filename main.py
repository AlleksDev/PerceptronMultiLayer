import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging (warnings e info)

from typing import List
from model_builder import ModelBuilder
from cross_validator import CrossValidator
from data.data_generator import generate_synthetic_dataset

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
    X_synthetic, y_synthetic = generate_synthetic_dataset(dataset_size=dataset_size, input_dim=input_dim)

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

