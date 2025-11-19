"""
supervisado.py

Entrena y evalúa un modelo SUPERVISADO
usando embeddings de user_embeddings_PCA.csv.
"""

from preproceso import cargar_datos_supervisado_desde_pca
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def main():
    # 1) Cargar datos procesados desde PCA
    X, y, df = cargar_datos_supervisado_desde_pca("user_embeddings_PCA.csv")

    # 2) Split estratificado train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3) Entrenar modelo supervisado
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 4) Evaluación
    preds = clf.predict(X_test)

    print("\n=========== RESULTADOS DEL MODELO SUPERVISADO ===========")
    print("Accuracy:", accuracy_score(y_test, preds))

    print("\nClassification report:")
    print(classification_report(y_test, preds))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()

