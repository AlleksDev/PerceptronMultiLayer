# Perceptron Multicapa y Validación Cruzada Dinámica

Este proyecto implementa una red neuronal de Perceptrón Multicapa (MLP) completamente construida de forma dinámica utilizando **TensorFlow 2.x**. Implementa el **Principio de Responsabilidad Única (SRP)** para separar la construcción del modelo, la recolección de los datos y el ciclo de la validación.

Destaca por integrar un **Sistema de Validación Cruzada (K-Fold) inverso personalizado**, permitiendo evaluar iterativamente los bloques en orden reverso (n, n-1, ..., 1) con un cálculo propio que pondera el error final sobre el tamaño total de la muestra combinada entre Entrenamiento y Validación.

## 🗂️ Estructura del Proyecto

* **`main.py`**: El punto de entrada (Entry Point). Inyecta las variables de configuración, unifica los módulos y ejecuta la validación maestra.
* **`model_builder.py`**: Aísla la lógica para construir la capa de entrada, las capas ocultas dinámicas requeridas por el usuario y la capa de predicción lineal continua junto con la compilación del optimizador y regresión (MSE).
* **`cross_validator.py`**: Recibe instancias vírgenes del `ModelBuilder` y gestiona tanto la partición de los bloqueos K-Fold Inverso como el cálculo del error ponderado.
* **`data_generator.py`**: Módulo puro encargado de la estadística sintética y generador de tensores `(X, y)` aleatorio.
* **`requirements.txt`**: Dependencias necesarias para ejecutar el sistema.

## 🛠️ Requisitos e Instalación

Para ejecutar este proyecto de Deep Learning se necesita Python 3.10+ y crear un entorno virtual para instalar las dependencias:

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

# 3. Instalar librerías
pip install -r requirements.txt
```

## 🚀 Uso del Proyecto

Una vez que tengas todo el ambiente listo, en tu terminal o IDE simplemente ejecuta:

```bash
python main.py
```

El script mostrará:
1. El informe de la topología (Sumario de la red dinámica).
2. Las iteraciones inversas de cada partición K-Fold, junto con la pérdida extraída tanto de Entrenamiento como de Validación.
3. El resultado global del Error Total del sistema empleando la fórmula de ponderación de tamaño analizada.

## 🧬 Configuración Rápida
Dentro de `main.py`, puedes configurar dinámicamente las capas, métricas o el tamaño del K-Fold ajustando en el bloque condicional principal:
```python
input_dim: int = 10
output_dim: int = 1
layers_config: List[int] = [64, 32, 16] # Modifica esto para añadir o cambiar capas.
activations: List[str] = ['relu', 'relu', 'tanh'] # Funciones de activación para cada capa intermedia.
dataset_size: int = 1000

# Bloques del K-Fold
n_splits: int = 5
```