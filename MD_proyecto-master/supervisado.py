"""
supervisado.py

Entrenamiento y evaluación de un modelo SUPERVISADO
a partir de user_embeddings_PCA.csv, usando la función
de preproceso cargar_datos_supervisado_desde_pca().

Este script NO toca tu flujo de K-Means. Es un módulo aparte.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preproceso import cargar_datos_supervisado_desde_pca


def train_test_split_supervisado(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Hace la división train/test estratificada para mantener
    las proporciones de clases.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def entrenar_modelo_supervisado(X_train, y_train):
    """
    Entrena un modelo supervisado sencillo.
    Usamos RandomForestClassifier como baseline robusto.
    """
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluar_modelo(clf, X_train, y_train, X_test, y_test):
    """
    Imprime métricas básicas en train y test.
    Más adelante podemos conectar esto con evaluacion.py.
    """
    # Predicciones
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print("\n================= MÉTRICAS EN TRAIN =================")
    print(f"Accuracy (train): {accuracy_score(y_train, y_train_pred):.4f}")

    print("\n================== MÉTRICAS EN TEST =================")
    print(f"Accuracy (test): {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification report (TEST):")
    print(classification_report(y_test, y_test_pred))

    print("\nMatriz de confusión (TEST):")
    print(confusion_matrix(y_test, y_test_pred))


def main():
    # 1) Cargar datos ya preprocesados desde PCA
    X, y, df = cargar_datos_supervisado_desde_pca("user_embeddings_PCA.csv")

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split_supervisado(X, y)

    # 3) Entrenar modelo supervisado
    clf = entrenar_modelo_supervisado(X_train, y_train)

    # 4) Evaluar modelo
    evaluar_modelo(clf, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

