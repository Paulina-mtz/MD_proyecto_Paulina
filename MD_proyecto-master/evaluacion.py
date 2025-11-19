# evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, adjusted_rand_score, normalized_mutual_info_score
import json
from proyecto import CustomKMeans, parse_embedding, load_kmeans_model
import warnings

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    rand_score,
    pairwise_distances
)
from sklearn.cluster import KMeans as SKKMeans
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings('ignore')


def _resolve_metric(distance_metric: str):
    """
    Devuelve métrica compatible con sklearn ('euclidean'|'cosine').
    """
    m = (distance_metric or "").lower()
    return m if m in {"euclidean", "cosine"} else "euclidean"


def cohesion_internal(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    Cohesión promedio (menor mejor): media de distancias intra-cluster por cluster.
    """
    clusters = np.unique(labels)
    if len(clusters) <= 1:
        return np.nan
    metric = _resolve_metric(metric)
    vals = []
    for c in clusters:
        Xc = X[labels == c]
        n = Xc.shape[0]
        if n <= 1:
            vals.append(0.0)
            continue
        D = pairwise_distances(Xc, Xc, metric=metric)
        mean_intra = (D.sum() - np.trace(D)) / (n * (n - 1))  # media sin diagonal
        vals.append(mean_intra)
    return float(np.mean(vals)) if vals else np.nan


def separability_internal(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    Separabilidad (mayor mejor): media de distancias entre centroides de clusters.
    """
    clusters = np.unique(labels)
    if len(clusters) <= 1:
        return np.nan
    metric = _resolve_metric(metric)
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in clusters])
    DC = pairwise_distances(centroids, centroids, metric=metric)
    k = len(clusters)
    upper_sum = DC.sum() - np.trace(DC)
    num_pairs = k * (k - 1)
    return float(upper_sum / num_pairs) if num_pairs > 0 else np.nan


def sse_within(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    SSE intra-cluster (menor mejor). Euclídea exacta; coseno aproximada con distancias coseno al centroide.
    """
    clusters = np.unique(labels)
    if len(clusters) <= 1:
        return np.nan
    metric = _resolve_metric(metric)
    total = 0.0
    for c in clusters:
        Xc = X[labels == c]
        if Xc.shape[0] == 0:
            continue
        mu = Xc.mean(axis=0, keepdims=True)
        if metric == "euclidean":
            residual = Xc - mu
            total += float(np.sum(residual * residual))
        else:
            Dc = pairwise_distances(Xc, mu, metric="cosine")
            total += float(np.sum(Dc ** 2))
    return total


def composite_index(X: np.ndarray, labels: np.ndarray, alpha: float, metric: str):
    """
    Índice compuesto (minimizar): J = α * Cohesión - (1-α) * Separabilidad
    """
    coh = cohesion_internal(X, labels, metric)
    sep = separability_internal(X, labels, metric)
    if np.isnan(coh) or np.isnan(sep):
        return np.nan, coh, sep
    J = alpha * coh - (1.0 - alpha) * sep
    return J, coh, sep

def external_scores(y_ref: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Rand": float(rand_score(y_ref, y_pred)),
        "Adjusted_Rand_Index": float(adjusted_rand_score(y_ref, y_pred)),
        "Normalized_Mutual_Info": float(normalized_mutual_info_score(y_ref, y_pred)),
    }


def build_external_clustering(X: np.ndarray, k: int, method: str = "sk_kmeans", random_state: int = 0) -> np.ndarray:
    method = (method or "").lower()
    if method == "sk_kmeans":
        km = SKKMeans(n_clusters=k, n_init=10, random_state=random_state)
        return km.fit_predict(X)
    elif method == "agg_ward":
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward", affinity="euclidean")
        return agg.fit_predict(X)
    elif method == "agg_complete":
        agg = AgglomerativeClustering(n_clusters=k, linkage="complete", affinity="euclidean")
        return agg.fit_predict(X)
    else:
        raise ValueError(f"Método externo desconocido: {method}")


class KMeansEvaluator:
    """
    Clase para evaluar el modelo K-Means comparando clusters con labels reales
    """

    def __init__(self, kmeans_model, X_train, y_train, X_test, y_test, train_phrases, test_phrases):
        self.kmeans = kmeans_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_phrases = train_phrases
        self.test_phrases = test_phrases

        # Predecir clusters
        self.train_clusters = kmeans_model.predict(X_train)
        self.test_clusters = kmeans_model.predict(X_test)

        
        self.metric = _resolve_metric(getattr(kmeans_model, "distance_metric", "euclidean"))

    def plot_class_to_cluster_matrix(self, data_type='train'):
        """
        Crea matriz de confusión entre clases reales y clusters

        Args:
            data_type: 'train' o 'test'
        """
        if data_type == 'train':
            true_labels = self.y_train
            pred_clusters = self.train_clusters
            title_suffix = ' (Train)'
        else:
            true_labels = self.y_test
            pred_clusters = self.test_clusters
            title_suffix = ' (Test)'

        # Crear matriz de confusión
        unique_labels = sorted(true_labels.unique())
        unique_clusters = sorted(np.unique(pred_clusters))

        # Crear matriz labels vs clusters
        matrix = np.zeros((len(unique_labels), len(unique_clusters)))

        for i, label in enumerate(unique_labels):
            for j, cluster in enumerate(unique_clusters):
                matrix[i, j] = np.sum((true_labels == label) & (pred_clusters == cluster))

        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix,
                    annot=True,
                    fmt='g',
                    xticklabels=[f'Cluster {c}' for c in unique_clusters],
                    yticklabels=[f'Class {l}' for l in unique_labels],
                    cmap='Blues')

        plt.xlabel('Clusters Asignados')
        plt.ylabel('Clases Reales')
        plt.title(f'Matriz Clase-a-Cluster{title_suffix}\n'
                  f'(K={self.kmeans.n_clusters}, Métrica={self.kmeans.distance_metric})')
        plt.tight_layout()
        plt.savefig(f'class_to_cluster_matrix_{data_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return matrix

    def calculate_cluster_metrics(self, data_type='train'):
        """
        Calcula métricas de evaluación para los clusters
        """
        if data_type == 'train':
            true_labels = self.y_train
            pred_clusters = self.train_clusters
            X = self.X_train
        else:
            true_labels = self.y_test
            pred_clusters = self.test_clusters
            X = self.X_test

        # Métricas de clustering (externas vs labels reales)
        ari = adjusted_rand_score(true_labels, pred_clusters)
        nmi = normalized_mutual_info_score(true_labels, pred_clusters)

        # Silhouette / Davies-Bouldin / Calinski-Harabasz
        try:
            sil = silhouette_score(X, pred_clusters, metric=self.metric)
        except Exception:
            sil = np.nan
        try:
            db = davies_bouldin_score(X, pred_clusters)
        except Exception:
            db = np.nan
        try:
            ch = calinski_harabasz_score(X, pred_clusters)
        except Exception:
            ch = np.nan

        # Cohesión / Separabilidad / SSE / Índice compuesto (α=0.6)
        J, coh, sep = composite_index(X, pred_clusters, alpha=0.6, metric=self.metric)
        sse = sse_within(X, pred_clusters, metric=self.metric)

        return {
            'Adjusted_Rand_Index': ari,
            'Normalized_Mutual_Info': nmi,
            'Silhouette': sil,
            'Davies_Bouldin': db,
            'Calinski_Harabasz': ch,
            'Cohesion': coh,
            'Separability': sep,
            'Composite_Index(alpha=0.6)': J,
            'SSE_within': sse,
            'Number_of_Clusters': len(np.unique(pred_clusters)),
            'Number_of_Classes': len(np.unique(true_labels))
        }

    def analyze_cluster_composition(self, data_type='train'):
        """
        Analiza la composición de cada cluster
        """
        if data_type == 'train':
            true_labels = self.y_train
            pred_clusters = self.train_clusters
            title_suffix = 'Train'
        else:
            true_labels = self.y_test
            pred_clusters = self.test_clusters
            title_suffix = 'Test'

        unique_clusters = sorted(np.unique(pred_clusters))
        unique_labels = sorted(true_labels.unique())

        print(f"\n{'=' * 60}")
        print(f"ANÁLISIS DE COMPOSICIÓN DE CLUSTERS - {title_suffix}")
        print(f"{'=' * 60}")

        results = {}

        for cluster in unique_clusters:
            cluster_mask = (pred_clusters == cluster)
            cluster_size = np.sum(cluster_mask)
            cluster_labels = true_labels[cluster_mask]

            if cluster_size > 0:
                label_counts = cluster_labels.value_counts()
                most_common_label = label_counts.index[0]
                most_common_count = label_counts.iloc[0]
                purity = most_common_count / cluster_size

                results[cluster] = {
                    'size': cluster_size,
                    'purity': purity,
                    'dominant_label': most_common_label,
                    'label_distribution': label_counts.to_dict()
                }

                print(f"\n📊 Cluster {cluster}:")
                print(f"   Tamaño: {cluster_size} instancias")
                print(f"   Pureza: {purity:.3f}")
                print(f"   Label dominante: {most_common_label}")
                print(f"   Distribución de labels: {label_counts.to_dict()}")
            else:
                print(f"\n📊 Cluster {cluster}: VACÍO")
                results[cluster] = {'size': 0, 'purity': 0, 'dominant_label': 'None', 'label_distribution': {}}

        return results

    def compare_with_external(self, method: str = "sk_kmeans", data_type='train', random_state: int = 0):
        if data_type == 'train':
            X = self.X_train
            y_pred = self.train_clusters
        else:
            X = self.X_test
            y_pred = self.test_clusters

        k = len(np.unique(y_pred))
        y_ext = build_external_clustering(X, k=k, method=method, random_state=random_state)
        scores = external_scores(y_ext, y_pred)
        print(f"   [EXTERNA] {data_type.upper()} vs {method}: "
              f"Rand={scores['Rand']:.4f} | ARI={scores['Adjusted_Rand_Index']:.4f} | NMI={scores['Normalized_Mutual_Info']:.4f}")
        return scores

    def print_detailed_comparison(self):
        """
        Imprime comparación detallada entre train y test
        """
        print(f"\n{'=' * 80}")
        print("EVALUACIÓN COMPLETA K-MEANS - COMPARACIÓN TRAIN vs TEST")
        print(f"{'=' * 80}")

        # Métricas para train
        train_metrics = self.calculate_cluster_metrics('train')
        print(f"\n📈 MÉTRICAS - TRAIN:")
        for metric, value in train_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")

        # Métricas para test
        test_metrics = self.calculate_cluster_metrics('test')
        print(f"\n📈 MÉTRICAS - TEST:")
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")

        # Análisis de composición
        self.analyze_cluster_composition('train')
        self.analyze_cluster_composition('test')

        print(f"\n{'-' * 80}")
        print("🔁 COMPARACIÓN EXTERNA (mismo K que tu modelo)")
        for method in ["sk_kmeans", "agg_ward", "agg_complete"]:
            self.compare_with_external(method=method, data_type='train', random_state=0)
            self.compare_with_external(method=method, data_type='test', random_state=0)

    def create_comprehensive_report(self):
        """
        Crea un reporte completo con visualizaciones
        """
        # Matrices clase-a-cluster
        train_matrix = self.plot_class_to_cluster_matrix('train')
        test_matrix = self.plot_class_to_cluster_matrix('test')

        # Métricas comparativas (tu flujo original + extras ya integrados)
        self.print_detailed_comparison()

        return {
            'train_metrics': self.calculate_cluster_metrics('train'),
            'test_metrics': self.calculate_cluster_metrics('test'),
        }


def load_and_prepare_data():
    """
    Carga y prepara los datos para evaluación
    """
    print("📂 Cargando datos...")

    # Cargar datos originales
    df = pd.read_csv('user_embeddings.csv')
    X = df['embedding']
    y = df['label']

    # Recrear el split (mismo random_state para consistencia)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir embeddings
    print("🔄 Parseando embeddings...")
    X_train_parsed = np.array([parse_embedding(embedding) for embedding in X_train])
    X_test_parsed = np.array([parse_embedding(embedding) for embedding in X_test])

    # Filtrar valores None
    valid_indices_train = [i for i, x in enumerate(X_train_parsed) if x is not None]
    valid_indices_test = [i for i, x in enumerate(X_test_parsed) if x is not None]

    X_train_clean = X_train_parsed[valid_indices_train]
    X_test_clean = X_test_parsed[valid_indices_test]
    y_train_clean = y_train.iloc[valid_indices_train].reset_index(drop=True)
    y_test_clean = y_test.iloc[valid_indices_test].reset_index(drop=True)

    # Frases
    train_phrases = df.iloc[X_train.index[valid_indices_train]]['frase'].reset_index(drop=True)
    test_phrases = df.iloc[X_test.index[valid_indices_test]]['frase'].reset_index(drop=True)

    print(f"✅ Datos cargados:")
    print(f"   - Train: {X_train_clean.shape[0]} muestras")
    print(f"   - Test: {X_test_clean.shape[0]} muestras")
    print(f"   - Clases únicas: {sorted(y.unique())}")

    return X_train_clean, y_train_clean, X_test_clean, y_test_clean, train_phrases, test_phrases


def main():
    """
    Función principal de evaluación
    """
    print("🎯 INICIANDO EVALUACIÓN COMPLETA DEL MODELO K-MEANS")
    print("=" * 60)

    # Cargar datos
    X_train, y_train, X_test, y_test, train_phrases, test_phrases = load_and_prepare_data()

    # Intentar cargar modelo guardado
    try:
        print("\n🔄 Cargando modelo guardado...")
        kmeans_model, _, _, _ = load_kmeans_model(
            'kmeans_model.json', X_train, y_train, train_phrases
        )
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        print("🔄 Entrenando nuevo modelo...")

        # Entrenar nuevo modelo si no existe
        kmeans_model = CustomKMeans(
            n_clusters=5,
            max_iters=100,
            random_state=42,
            distance_metric='cosine'
        )
        kmeans_model.fit(X_train)
        print("✅ Nuevo modelo entrenado")

    # Crear evaluador
    print("\n📊 Creando evaluador...")
    evaluator = KMeansEvaluator(
        kmeans_model, X_train, y_train, X_test, y_test, train_phrases, test_phrases
    )

    # Ejecutar evaluación completa
    print("\n📈 Ejecutando evaluación completa...")
    results = evaluator.create_comprehensive_report()

    # Resumen final (mantengo tu salida original y añado algunas claves internas)
    print(f"\n{'=' * 80}")
    print("🎯 RESUMEN FINAL DE EVALUACIÓN")
    print(f"{'=' * 80}")

    print(f"\n🔍 CONFIGURACIÓN DEL MODELO:")
    print(f"   - Número de clusters (K): {kmeans_model.n_clusters}")
    print(f"   - Métrica de distancia: {kmeans_model.distance_metric}")
    print(f"   - Inercia final: {getattr(kmeans_model, 'inertia_', np.nan):.2f}")
    print(f"   - Iteraciones: {getattr(kmeans_model, 'n_iter_', np.nan)}")

    print(f"\n📊 RESULTADOS CLAVE:")
    print(f"   - ARI Train: {results['train_metrics']['Adjusted_Rand_Index']:.4f}")
    print(f"   - ARI Test:  {results['test_metrics']['Adjusted_Rand_Index']:.4f}")
    print(f"   - NMI Train: {results['train_metrics']['Normalized_Mutual_Info']:.4f}")
    print(f"   - NMI Test:  {results['test_metrics']['Normalized_Mutual_Info']:.4f}")

    print(f"   - Silhouette Train: {results['train_metrics'].get('Silhouette', float('nan')):.4f}")
    print(f"   - Silhouette Test:  {results['test_metrics'].get('Silhouette', float('nan')):.4f}")
    print(f"   - Cohesion Train:   {results['train_metrics'].get('Cohesion', float('nan')):.4f}")
    print(f"   - Separability Train:{results['train_metrics'].get('Separability', float('nan')):.4f}")
    print(f"   - SSE Train:        {results['train_metrics'].get('SSE_within', float('nan')):.2f}")

    print(f"\n💾 ARCHIVOS GENERADOS:")
    print(f"   - class_to_cluster_matrix_train.png")
    print(f"   - class_to_cluster_matrix_test.png")

    print(f"\n✅ EVALUACIÓN COMPLETADA")


if __name__ == "__main__":
    main()
