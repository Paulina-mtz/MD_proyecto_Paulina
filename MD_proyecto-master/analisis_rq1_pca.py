"""
analisis_rq1_pca.py

Análisis de la separabilidad de las clases en el espacio PCA
para responder la RQ1:

"¿Cómo afecta la reducción de dimensionalidad mediante PCA a la
separabilidad de las clases del dataset, evaluada desde un punto
de vista geométrico y estadístico?"

Este script:
    - Carga X, y desde user_embeddings_PCA.csv
      usando preproceso.cargar_datos_supervisado_desde_pca
    - Calcula silhouette global y por clase
    - Calcula distancias entre centroides de clase
    - Calcula dispersión intra-clase
    - (Opcional) genera una proyección t-SNE 2D coloreada por clase
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE

from preproceso import cargar_datos_supervisado_desde_pca


def calcular_silhouette_por_clase(X: pd.DataFrame, y: pd.Series):
    """
    Devuelve:
        - silhouette_global (float)
        - silhouette_por_clase (Series: Label -> media de silhouette)
    """
    X_np = X.values
    y_np = y.to_numpy()

    print("[INFO] Calculando silhouette global...")
    sil_global = silhouette_score(X_np, y_np, metric="euclidean")
    print(f"[OK] Silhouette global: {sil_global:.4f}")

    print("[INFO] Calculando silhouette individual y agrupando por clase...")
    sil_samples = silhouette_samples(X_np, y_np, metric="euclidean")
    df_sil = pd.DataFrame({"Label": y_np, "silhouette": sil_samples})

    sil_por_clase = df_sil.groupby("Label")["silhouette"].mean().sort_values(ascending=False)
    print("\n[RESULTADO] Silhouette medio por clase:")
    print(sil_por_clase)

    return sil_global, sil_por_clase


def calcular_geometria_clases(X: pd.DataFrame, y: pd.Series):
    """
    Calcula:
        - centroides por clase (dict: Label -> vector numpy)
        - distancia media intra-clase (dict: Label -> float)
        - matriz de distancias entre centroides (DataFrame)
    """
    X_np = X.values
    y_np = y.to_numpy()

    clases = sorted(np.unique(y_np))
    centroids = {}
    intra_dist = {}

    print("\n[INFO] Calculando centroides y dispersión intra-clase...")
    for c in clases:
        mask = (y_np == c)
        X_c = X_np[mask]
        centroide = X_c.mean(axis=0)
        centroids[c] = centroide
        # Distancia media de los puntos de la clase a su centroide
        distancias = np.linalg.norm(X_c - centroide, axis=1)
        intra_dist[c] = distancias.mean()

    print("\n[RESULTADO] Dispersión intra-clase (distancia media al centroide):")
    for c in clases:
        print(f"  {c:10s}: {intra_dist[c]:.4f}")

    # Matriz de distancias entre centroides
    print("\n[INFO] Calculando distancias entre centroides de clase...")
    n = len(clases)
    dist_matrix = np.zeros((n, n))
    for i, ci in enumerate(clases):
        for j, cj in enumerate(clases):
            dist = np.linalg.norm(centroids[ci] - centroids[cj])
            dist_matrix[i, j] = dist

    dist_df = pd.DataFrame(dist_matrix, index=clases, columns=clases)

    print("\n[RESULTADO] Matriz de distancias entre centroides (primeras filas):")
    print(dist_df)

    return centroids, intra_dist, dist_df


def generar_tsne_2d(X: pd.DataFrame, y: pd.Series, out_png: str = "rq1_tsne_pca_labels.png", max_points: int = 5000):
    """
    Genera un t-SNE 2D para visualizar la separabilidad de las clases
    en el espacio PCA. Submuestrea hasta max_points puntos para que sea
    computacionalmente manejable.

    Guarda la figura en out_png.
    """
    X_np = X.values
    y_np = y.to_numpy()

    n = X_np.shape[0]
    if n > max_points:
        print(f"[INFO] Submuestreando {max_points} de {n} puntos para t-SNE...")
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        X_sub = X_np[idx]
        y_sub = y_np[idx]
    else:
        X_sub = X_np
        y_sub = y_np

    print("[INFO] Ejecutando t-SNE 2D (puede tardar un rato)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(X_sub)

    # Asignar un color distinto por clase
    clases = sorted(np.unique(y_sub))
    color_map = {c: i for i, c in enumerate(clases)}
    colors = [color_map[c] for c in y_sub]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=5, alpha=0.7)
    plt.title("t-SNE de embeddings PCA coloreado por clase (RQ1)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # Leyenda manual
    # Se toma un punto representativo por clase
    handles = []
    labels = []
    for c in clases:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=6))
        labels.append(c)
    plt.legend(handles, labels, title="Clase", loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Figura t-SNE guardada en: {out_png}")


def main():
    # 1) Cargar datos desde PCA
    X, y, df = cargar_datos_supervisado_desde_pca("user_embeddings_PCA.csv")

    # 2) Silhouette global y por clase
    sil_global, sil_por_clase = calcular_silhouette_por_clase(X, y)

    # 3) Geometría de clases: centroides, intra-clase, distancias
    centroids, intra_dist, dist_df = calcular_geometria_clases(X, y)

    # 4) Guardar resultados en CSV para el informe
    print("\n[INFO] Guardando resultados numéricos en CSV...")
    sil_por_clase.to_frame("silhouette_mean").to_csv("rq1_silhouette_por_clase.csv")
    pd.Series(intra_dist, name="intra_mean_distance").to_csv("rq1_intra_clase_dist.csv")
    dist_df.to_csv("rq1_centroid_distances.csv")
    print("[OK] Guardado: rq1_silhouette_por_clase.csv")
    print("[OK] Guardado: rq1_intra_clase_dist.csv")
    print("[OK] Guardado: rq1_centroid_distances.csv")

    # 5) t-SNE 2D para visualización (opcional pero MUY útil para el reporte)
    generar_tsne_2d(X, y, out_png="rq1_tsne_pca_labels.png", max_points=5000)

    print("\n[FIN] Análisis RQ1 completado.")


if __name__ == "__main__":
    main()
