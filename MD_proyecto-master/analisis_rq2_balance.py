"""
analisis_rq2_balance.py

Análisis del impacto del desbalance de clases y comparación de técnicas
de reequilibrio para responder la RQ2:

"¿En qué medida el desbalance de clases afecta al desempeño del modelo 
supervisado, y qué técnicas de reequilibrio producen mejores mejoras 
en F1 macro y recall por clase?"

Este script prueba:
    - Baseline (sin balanceo)
    - class_weight='balanced'
    - Oversampling (RandomOverSampler)
    - Undersampling (RandomUnderSampler)
    - SMOTE (si está disponible)

Y guarda:
    - rq2_resultados_balance.csv (tabla de métricas)
    - rq2_f1_macro_por_metodo.png (gráfica de barras de F1 Macro)
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preproceso import cargar_datos_supervisado_desde_pca

# Opcionales (instalar imbalanced-learn para usar estas técnicas)
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE

    USE_IMBLEARN = True
except ImportError:
    print("[AVISO] imblearn no está instalado. SMOTE, oversampling y undersampling desactivados.")
    USE_IMBLEARN = False


def entrenar_modelo(X_train, y_train, class_weight=None):
    """
    Entrena un RandomForest con los datos dados y, opcionalmente,
    con pesos de clase.
    """
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluar_modelo(nombre, clf, X_test, y_test):
    """
    Calcula métricas básicas (Accuracy y F1 Macro) e imprime
    también el classification_report y la matriz de confusión.
    """
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")

    print(f"\n========== {nombre} ==========")
    print("Accuracy:", acc)
    print("F1 Macro:", f1_macro)
    print("\nReporte:\n", classification_report(y_test, preds))
    print("Matriz de confusión:\n", confusion_matrix(y_test, preds))

    return {
        "Metodo": nombre,
        "Accuracy": acc,
        "F1_macro": f1_macro,
    }


def generar_grafica_f1(df_resultados: pd.DataFrame, out_png: str = "rq2_f1_macro_por_metodo.png"):
    """
    Genera una gráfica de barras con el F1 Macro por método.
    """
    print("\n[INFO] Generando gráfica de F1 Macro por método...")

    plt.figure(figsize=(8, 5))
    x = range(len(df_resultados))
    plt.bar(x, df_resultados["F1_macro"])
    plt.xticks(x, df_resultados["Metodo"], rotation=20, ha="right")
    plt.ylabel("F1 Macro")
    plt.title("Comparación de F1 Macro por técnica de reequilibrio (RQ2)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] Gráfica guardada en: {out_png}")


def main():
    # 1) Cargar datos
    X, y, df = cargar_datos_supervisado_desde_pca("user_embeddings_PCA.csv")

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    resultados = []

    # ---- Baseline ----
    clf_base = entrenar_modelo(X_train, y_train, class_weight=None)
    resultados.append(evaluar_modelo("Baseline (sin balanceo)", clf_base, X_test, y_test))

    # ---- class_weight='balanced' ----
    clf_bal = entrenar_modelo(X_train, y_train, class_weight="balanced")
    resultados.append(evaluar_modelo("class_weight='balanced'", clf_bal, X_test, y_test))

    if USE_IMBLEARN:
        # ---- Oversampling ----
        print("\n[INFO] Aplicando Oversampling...")
        ros = RandomOverSampler(random_state=42)
        X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
        clf_over = entrenar_modelo(X_train_over, y_train_over)
        resultados.append(
            evaluar_modelo("Oversampling (RandomOverSampler)", clf_over, X_test, y_test)
        )

        # ---- Undersampling ----
        print("\n[INFO] Aplicando Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
        clf_under = entrenar_modelo(X_train_under, y_train_under)
        resultados.append(
            evaluar_modelo("Undersampling (RandomUnderSampler)", clf_under, X_test, y_test)
        )

        # ---- SMOTE ----
        print("\n[INFO] Aplicando SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        clf_smote = entrenar_modelo(X_train_smote, y_train_smote)
        resultados.append(evaluar_modelo("SMOTE", clf_smote, X_test, y_test))

    # 3) Guardar tabla de resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("rq2_resultados_balance.csv", index=False)
    print("\n[OK] Resultados guardados en rq2_resultados_balance.csv")

    # 4) Generar gráfica de F1 Macro por método
    generar_grafica_f1(df_resultados, out_png="rq2_f1_macro_por_metodo.png")

    print("\n[FIN] Análisis RQ2 completado.")


if __name__ == "__main__":
    main()
