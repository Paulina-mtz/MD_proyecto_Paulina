import pandas as pd
import ast
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent

PATH_LABELS = BASE / "MD_proyecto-master" / "500_Reddit_users_posts_labels.csv"
PATH_EMBS   = BASE / "MD_proyecto-master" / "user_embeddings_PCA.csv"


def cargar_datos():
    # === 1. Cargar ambos CSV ===
    labels = pd.read_csv(PATH_LABELS)
    embs   = pd.read_csv(PATH_EMBS)

    # --- Normalizar columna User para que coincida ---
    labels['User'] = labels['User'].str.replace("user-", "").astype(int)

    # --- Convertir embedding (string) → lista ---
    embs['embedding'] = embs['embedding'].apply(ast.literal_eval)

    # --- Unir por User ---
    df = labels.merge(embs[['User', 'embedding']], on='User', how='inner')

    print("Dataset combinado:", df.shape)

    # --- Expandir lista de embedding a columnas numéricas ---
    X = pd.DataFrame(df['embedding'].tolist())

    # --- Target ---
    y = df['Label']

    return X, y


def main():
    X, y = cargar_datos()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
