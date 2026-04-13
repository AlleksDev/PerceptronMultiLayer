import numpy as np
from sklearn.model_selection import KFold
from typing import Dict
from src.model_builder import ModelBuilder

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
        '''
        # --- CÓDIGO ANTERIOR (Validación Cruzada Estándar) ---
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_losses = []
        fold_maes = []
        
        print(f"\\n--- Iniciando Validación Cruzada ({self.n_splits} Folds) ---")
        
        # Iteración explícita sobre splits
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            print(f"\\n[Fold {fold}/{self.n_splits}]")
            
            # Segmentar datos usando los índices provistos por KFold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Recrear el modelo para limpiar pesos y evitar la filtración de conocimiento transversal
            model = self.builder.build()
            
            print(f"  -> Entrenando: {len(train_idx)} muestras | Validando: {len(val_idx)} muestras")
            
            # Entrenamiento (verbose=0)
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Evaluación final del fold
            results = model.evaluate(X_val, y_val, verbose=0)
            
            # Asegurar captura del loss y el mae
            loss_val = results[0] if isinstance(results, list) else results
            mae_val = results[1] if isinstance(results, list) and len(results) > 1 else 0.0
            
            print(f"  -> MSE (Loss): {loss_val:.4f}  |  MAE: {mae_val:.4f}")
            
            fold_losses.append(loss_val)
            fold_maes.append(mae_val)
            
        # Calcular estadísticos generales
        avg_loss = float(np.mean(fold_losses))
        avg_mae = float(np.mean(fold_maes))
        
        print(f"\\n--- Resumen Final del K-Fold ({self.n_splits} splits) ---")
        print(f"Promedio Loss (MSE): {avg_loss:.4f}  (+/- {np.std(fold_losses):.4f})")
        print(f"Promedio MAE:        {avg_mae:.4f}  (+/- {np.std(fold_maes):.4f})")
        
        return {
            "mean_loss": avg_loss,
            "mean_mae": avg_mae
        }
        '''

        # --- NUEVA LÓGICA: Validación Inversa y Error Ponderado ---
        print(f"\n--- Iniciando Validación Cruzada Inversa ({self.n_splits} Folds) ---")
        
        # Generar los índices totales del dataset
        indices = np.arange(len(X))
        
        # Partición matemática exacta usando np.array_split para crear n_splits bloques
        splits = np.array_split(indices, self.n_splits)
        
        fold_val_losses = []
        fold_train_losses = []
        
        # Variables para rastrear el mejor modelo según el error ponderado del fold
        best_fold_error = float('inf')
        best_history = None

        # Contadores globales de observaciones
        total_obs_train = 0
        total_obs_val = 0
        
        # Iteración inversa: recorremos los bloques desde n_splits-1 hasta 0
        # range(start, stop, step)
        for iteration, i in enumerate(range(self.n_splits - 1, -1, -1), start=1):
            val_idx = splits[i]
            
            # Generar train extraiendo todos los bloques restantes
            train_idx = np.concatenate([splits[j] for j in range(self.n_splits) if j != i])
            
            print(f"\n[Iteración {iteration} | Bloque de Validación: {i + 1}/{self.n_splits}]")
            
            # Segmentar los datos
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            n_train = len(train_idx)
            n_val = len(val_idx)
            
            # Acumular conteos para fórmula ponderada final
            total_obs_train += n_train
            total_obs_val += n_val
            
            model = self.builder.build()
            
            print(f"  -> Entrenando: {n_train} muestras | Validando: {n_val} muestras")
            
            # Entrenamiento silencioso (verbose=0)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Evaluación separada de Train y Val para cálculo de Error_Total
            train_results = model.evaluate(X_train, y_train, verbose=0)
            val_results = model.evaluate(X_val, y_val, verbose=0)
            
            loss_train = train_results[0] if isinstance(train_results, list) else train_results
            loss_val = val_results[0] if isinstance(val_results, list) else val_results
            
            print(f"  -> Train MSE (Loss): {loss_train:.4f} | Val MSE (Loss): {loss_val:.4f}")
            
            # Calcular error ponderado del fold (total_obs_train_fold * loss_train + total_obs_val_fold * loss_val) / total_fold_obs
            fold_total_error = (n_train * loss_train + n_val * loss_val) / (n_train + n_val)
            print(f"  -> Error Total (Ponderado) en Fold {iteration}: {fold_total_error:.4f}")
            
            if fold_total_error < best_fold_error:
                best_fold_error = fold_total_error
                best_history = history.history
            
            fold_train_losses.append(loss_train)
            fold_val_losses.append(loss_val)
            
        # Calcular los promedios solicitados
        mean_error_train = float(np.mean(fold_train_losses))
        mean_error_val = float(np.mean(fold_val_losses))
        
        # Tamaño total acumulado = obs totales de todas iteraciones 
        total_dataset_size_accumulated = total_obs_train + total_obs_val
        
        # --- CÁLCULO FÓRMULA PERSONALIZADA --- 
        # Error_Total = (total_obs_train * mean_error_train + total_obs_val * mean_error_val) / total_dataset_size
        error_total_sistema = (
            (total_obs_train * mean_error_train) + (total_obs_val * mean_error_val)
        ) / total_dataset_size_accumulated
        
        print(f"\n--- Resumen Final validación Inversa ({self.n_splits} splits) ---")
        print(f"Promedio Train Loss (MSE): {mean_error_train:.4f}")
        print(f"Promedio Val Loss (MSE):   {mean_error_val:.4f}")
        print(f"Observaciones Totales Acumuladas - Train: {total_obs_train}, Val: {total_obs_val}")
        print(f"\n[Desglose del Error Ponderado Personalizado]")
        print(f"Fórmula: (({total_obs_train} * {mean_error_train:.4f}) + ({total_obs_val} * {mean_error_val:.4f})) / {total_dataset_size_accumulated}")
        print(f"=> Error Total del Sistema: {error_total_sistema:.4f}")
        
        return {
            "mean_loss": mean_error_val, # Compatible con implementaciones antiguas
            "error_total": error_total_sistema,
            "best_history": best_history
        }
