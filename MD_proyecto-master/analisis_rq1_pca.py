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
    - Genera:
        * Curve tipo scree (varianza por componente PCA)
        * Heatmap de distancias entre centroides
        * t-SNE 2D coloreado por clase
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


def generar_tsne_2d(
    X: pd.DataFrame,
    y: pd.Series,
    out_png: str = "rq1_tsne_pca_labels.png",
    max_points: int = 5000,
):
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
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=5, alpha=0.7)
    plt.title("t-SNE de embeddings PCA coloreado por clase (RQ1)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # Leyenda manual
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


def generar_scree_plot(X: pd.DataFrame, out_png: str = "rq1_scree_pca.png"):
    """
    Genera una gráfica tipo 'scree plot' usando la varianza de cada
    componente del espacio PCA ya reducido.

    Nota: no es la varianza explicada directa de sklearn.PCA,
    pero es proporcional a los eigenvalores si los datos están
    centrados, por lo que es válida para interpretar la importancia
    relativa de cada componente.
    """
    print("[INFO] Calculando varianza por componente (scree plot)...")
    # Varianza por columna (componente)
    variances = X.var(axis=0).values
    n_comp = len(variances)
    comps = np.arange(1, n_comp + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(comps, variances, marker="o")
    plt.title("Varianza por componente PCA (Scree-like plot)")
    plt.xlabel("Componente principal")
    plt.ylabel("Varianza")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Scree plot guardado en: {out_png}")


def generar_heatmap_centroides(
    dist_df: pd.DataFrame,
    out_png: str = "rq1_centroid_distances_heatmap.png",
):
    """
    Genera un heatmap de la matriz de distancias entre centroides de clase.
    """
    print("[INFO] Generando heatmap de distancias entre centroides...")
    plt.figure(figsize=(6, 5))
    im = plt.imshow(dist_df.values, cmap="viridis")

    plt.xticks(ticks=np.arange(len(dist_df.columns)), labels=dist_df.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(dist_df.index)), labels=dist_df.index)

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Distancia euclídea")
    plt.title("Heatmap de distancias entre centroides de clase (RQ1)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Heatmap de centroides guardado en: {out_png}")


def main():
    # 1) Cargar datos desde PCA
    X, y, df = cargar_datos_supervisado_desde_pca("user_embeddings_PCA.csv")

    # 2) Silhouette global y por clase
    sil_global, sil_por_clase = calcular_silhouette_por_clase(X, y)

    # 3) Geometría de clases: centroides, intra-clase, distancias
    centroids, intra_dist, dist_df = calcular_geometria_clases(X, y)

    # 4) Guardar resultados numéricos en CSV para el informe
    print("\n[INFO] Guardando resultados numéricos en CSV...")
    sil_por_clase.to_frame("silhouette_mean").to_csv("rq1_silhouette_por_clase.csv")
    pd.Series(intra_dist, name="intra_mean_distance").to_csv("rq1_intra_clase_dist.csv")
    dist_df.to_csv("rq1_centroid_distances.csv")
    print("[OK] Guardado: rq1_silhouette_por_clase.csv")
    print("[OK] Guardado: rq1_intra_clase_dist.csv")
    print("[OK] Guardado: rq1_centroid_distances.csv")

    # 5) Scree-like plot (varianza por componente PCA)
    generar_scree_plot(X, out_png="rq1_scree_pca.png")

    # 6) Heatmap de distancias entre centroides
    generar_heatmap_centroides(dist_df, out_png="rq1_centroid_distances_heatmap.png")

    # 7) t-SNE 2D para visualización (opcional pero MUY útil para el reporte)
    generar_tsne_2d(X, y, out_png="rq1_tsne_pca_labels.png", max_points=5000)

    print("\n[FIN] Análisis RQ1 completado.")


if __name__ == "__main__":
    main()
