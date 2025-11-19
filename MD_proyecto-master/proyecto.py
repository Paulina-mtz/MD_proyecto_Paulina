import random
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List
import time
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

## SPLIT ##
df = pd.read_csv('user_embeddings.csv')
X = df['embedding']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## implementacion del algoritmo K-MEANS ##

class CustomKMeans:
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4,
                 random_state: Optional[int] = None,
                 distance_metric: str = 'cosine'):
        """
        Implementación personalizada de K-Means

        Args:
            n_clusters: Número de clusters (k)
            max_iters: Máximo número de iteraciones
            tol: Tolerancia para convergencia
            random_state: Semilla para reproducibilidad
            distance_metric: Métrica de distancia ('cosine' o 'euclidean')
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Inicializa los centroides usando el metodo K-means++
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Primer centroide: elegir aleatoriamente
        first_idx = random.randint(0, n_samples - 1)
        centroids[0] = X[first_idx]

        # Para los siguientes centroides
        for i in range(1, self.n_clusters):
            # Calcular distancias al centroide más cercano para cada punto
            distances = self._compute_distances_to_centroids(X, centroids[:i])
            min_distances = np.min(distances, axis=1)

            # Probabilidad proporcional al cuadrado de la distancia
            probabilities = min_distances ** 2
            probabilities /= np.sum(probabilities)

            # Elegir siguiente centroide
            cumulative_probs = np.cumsum(probabilities)
            r = random.random()
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids[i] = X[next_idx]

        return centroids

    def _compute_distances_to_centroids(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calcula distancias entre puntos y centroides
        """
        n_samples = X.shape[0]
        n_centroids = centroids.shape[0]
        distances = np.zeros((n_samples, n_centroids))

        if self.distance_metric == 'cosine':
            # Normalizar vectores para cálculo de coseno
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
            centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

            # Distancia coseno = 1 - similitud coseno
            for i in range(n_centroids):
                cosine_similarity = np.dot(X_norm, centroids_norm[i])
                distances[:, i] = 1 - cosine_similarity

        else:
            for i in range(n_centroids):
                # Distancia euclidiana al cuadrado
                diff = X - centroids[i]
                distances[:, i] = np.sum(diff ** 2, axis=1)

        return distances

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Asigna cada punto al cluster más cercano
        """
        distances = self._compute_distances_to_centroids(X, centroids)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Actualiza los centroides calculando la media de cada cluster
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            # Puntos pertenecientes al cluster i
            cluster_points = X[labels == i]

            if len(cluster_points) > 0:
                if self.distance_metric == 'cosine':
                    # Para distancia coseno, normalizamos los centroides
                    centroid_mean = np.mean(cluster_points, axis=0)
                    centroid_norm = np.linalg.norm(centroid_mean)
                    if centroid_norm > 0:
                        new_centroids[i] = centroid_mean / centroid_norm
                    else:
                        new_centroids[i] = centroid_mean
                else:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # Si un cluster está vacío, reinicializar con un punto aleatorio
                new_centroids[i] = X[random.randint(0, X.shape[0] - 1)]

        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calcula la inercia (suma de distancias al cuadrado)
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                if self.distance_metric == 'cosine':

                    distances = self._compute_distances_to_centroids(cluster_points, centroids[i:i + 1])
                    inertia += np.sum(distances)
                else:
                    distances = np.sum((cluster_points - centroids[i]) ** 2)
                    inertia += distances
        return inertia

    def fit(self, X: np.ndarray) -> 'CustomKMeans':
        """
        Entrena el modelo K-Means con los datos X
        """
        X = np.array(X, dtype=np.float64)

        print(
            f"🚀 Iniciando K-Means con k={self.n_clusters} y métrica={self.distance_metric}")
        print(f"📊 Datos: {X.shape[0]} muestras, {X.shape[1]} características")

        # Inicializar centroides
        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iters):
            # Paso 1: Asignar clusters
            self.labels = self._assign_clusters(X, self.centroids)

            # Paso 2: Actualizar centroides
            new_centroids = self._update_centroids(X, self.labels)

            # Paso 3: Calcular cambio en centroides
            centroid_shift = np.sqrt(np.sum((self.centroids - new_centroids) ** 2, axis=1)).mean()

            # Paso 4: Actualizar centroides
            self.centroids = new_centroids

            # Paso 5: Calcular inercia
            self.inertia_ = self._compute_inertia(X, self.labels, self.centroids)
            self.n_iter_ = iteration + 1

            # Mostrar progreso cada 10 iteraciones
            if iteration % 10 == 0:
                print(f"   Iteración {iteration:3d} - Inercia: {self.inertia_:12.2f} - Shift: {centroid_shift:.6f}")

            # Verificar convergencia
            if centroid_shift < self.tol:
                print(f"✅ Convergencia alcanzada en iteración {iteration + 1}")
                break

        print(f"📊 Inercia final: {self.inertia_:.2f}")
        print(f"🔄 Iteraciones totales: {self.n_iter_}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice los clusters para nuevos datos
        """
        X = np.array(X, dtype=np.float64)
        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Entrena y predice en un solo paso
        """
        self.fit(X)
        return self.labels


class KMeansEvaluator:
    """
    Clase para evaluar y analizar resultados de K-Means
    """
    @staticmethod
    def calculate_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula el Silhouette Score usando sklearn
        """
        # Usar una muestra más pequeña para hacerlo manejable
        if len(X) > 10000:
            # Para datasets grandes, muestrear
            indices = np.random.choice(len(X), size=10000, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
            return silhouette_score(X_sample, labels_sample,
                                    metric='cosine')
        else:
            return silhouette_score(X, labels, metric='cosine')

    @staticmethod
    def find_optimal_k(X: np.ndarray, k_range: range,
                       max_iters: int = 50) -> dict:
        """
        Encuentra el k óptimo usando el metodo del codo
        """
        inertias = {}
        silhouette_scores = {}

        print("🔍 Buscando k óptimo...")
        for k in k_range:
            print(f"   Probando k={k}...")
            kmeans = CustomKMeans(n_clusters=k, max_iters=max_iters, random_state=42,
                                  # EDITADO: Usar max_iters reducido
                                  distance_metric='cosine')
            kmeans.fit(X)

            inertias[k] = kmeans.inertia_

            # Calcular silhouette score (más rápido)
            try:
                silhouette_scores[k] = KMeansEvaluator.calculate_silhouette_score(X, kmeans.labels)
                print(f"      ✅ Silhouette: {silhouette_scores[k]:.4f}")  #
            except Exception as e:
                print(f"      ⚠️  Error calculando silhouette: {e}")
                silhouette_scores[k] = -1

        return {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }


# Primero, necesitamos convertir los embeddings de string a arrays numpy
def parse_embedding(embedding_str):
    """Convierte el string del embedding a array numpy"""
    try:
        # Limpiar el string y convertir a lista de floats
        embedding_str = embedding_str.strip('[]')
        embedding_list = [float(x) for x in embedding_str.split(',')]
        return np.array(embedding_list)
    except Exception as e:
        print(f"Error parsing embedding: {e}")
        return None


# Convertir X_train y X_test
print("🔄 Convirtiendo embeddings...")
X_train_parsed = np.array([parse_embedding(embedding) for embedding in X_train])
X_test_parsed = np.array([parse_embedding(embedding) for embedding in X_test])

# Filtrar valores None (si los hay)
valid_indices_train = [i for i, x in enumerate(X_train_parsed) if x is not None]
valid_indices_test = [i for i, x in enumerate(X_test_parsed) if x is not None]

X_train_clean = X_train_parsed[valid_indices_train]
X_test_clean = X_test_parsed[valid_indices_test]
y_train_clean = y_train.iloc[valid_indices_train].reset_index(drop=True)
y_test_clean = y_test.iloc[valid_indices_test].reset_index(drop=True)

# Mantener también las frases originales para train y test
train_phrases = df.iloc[X_train.index[valid_indices_train]]['frase'].reset_index(drop=True)
test_phrases = df.iloc[X_test.index[valid_indices_test]]['frase'].reset_index(drop=True)

## EJECUTAR K-MEANS CON X-TRAIN ##

print("\n🔍 BUSCANDO K ÓPTIMO...")
k_range = range(5, 15)  # Probamos k desde 5 hasta 15
optimal_k_results = KMeansEvaluator.find_optimal_k(X_train_clean, k_range, max_iters=50)

# Mostrar resultados
print("\n📊 RESULTADOS BÚSQUEDA K ÓPTIMO:")
print("K\tInercia\t\tSilhouette")
for k in k_range:
    print(f"{k}\t{optimal_k_results['inertias'][k]:.2f}\t\t{optimal_k_results['silhouette_scores'][k]:.4f}")

# Encontrar k óptimo basado en silhouette score
best_k_silhouette = max(optimal_k_results['silhouette_scores'],
                        key=optimal_k_results['silhouette_scores'].get)
best_silhouette = optimal_k_results['silhouette_scores'][best_k_silhouette]

print(f"\n✅ K ÓPTIMO ENCONTRADO:")
print(f"   - Por Silhouette Score: k={best_k_silhouette} (score: {best_silhouette:.4f})")

# También podemos usar el metodo del codo visualmente
inertias = [optimal_k_results['inertias'][k] for k in k_range]
silhouettes = [optimal_k_results['silhouette_scores'][k] for k in k_range]

# Elegir k basado en los resultados (puedes ajustar esta lógica)
chosen_k = best_k_silhouette  # o elegir manualmente basado en los resultados
print(f"   - K ELEGIDO PARA ENTRENAMIENTO: {chosen_k}")
# Entrenar modelo final con k optimo
print(f"\n🚀 Entrenando modelo final con k={chosen_k} y distancia COSENO...")
kmeans = CustomKMeans(n_clusters=chosen_k, max_iters=100, random_state=42, distance_metric='cosine')
kmeans.fit(X_train_clean)

train_clusters = kmeans.labels
test_clusters = kmeans.predict(X_test_clean)

print(f"✅ Entrenamiento completado:")
print(f"   - Clusters en train: {np.unique(train_clusters)}")
print(f"   - Clusters en test: {np.unique(test_clusters)}")


## GUARDAR EL MODELO EN JSON ##
def save_kmeans_model(kmeans_model, filename):
    """Guarda el modelo K-Means en formato JSON"""
    model_data = {
        'n_clusters': kmeans_model.n_clusters,
        'centroids': kmeans_model.centroids.tolist(),
        'inertia': kmeans_model.inertia_,
        'n_iter': kmeans_model.n_iter_,
        'distance_metric': kmeans_model.distance_metric
    }

    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"💾 Modelo guardado en: {filename}")


# Guardar el modelo
save_kmeans_model(kmeans, 'kmeans_model.json')


## FUNCIÓN PARA ENCONTRAR LAS 5 INSTANCIAS MÁS CERCANAS ##
def find_nearest_instances(new_instance, training_data, training_labels, training_phrases, n_neighbors=5):
    """
    Encuentra las n instancias más cercanas en el conjunto de entrenamiento

    Args:
        new_instance: Nueva instancia a clasificar
        training_data: Datos de entrenamiento
        training_labels: Etiquetas originales de entrenamiento
        training_phrases: Frases originales de entrenamiento
        n_neighbors: Número de vecinos a encontrar

    Returns:
        Lista con las instancias más cercanas y sus distancias
    """

    distances = cosine_distances([new_instance], training_data)[0]

    # Obtener índices de los n_neighbors más cercanos
    nearest_indices = np.argsort(distances)[:n_neighbors]

    # Recolectar información de los vecinos más cercanos
    nearest_instances = []
    for idx in nearest_indices:
        instance_info = {
            'index': int(idx),
            'distance': float(distances[idx]),
            'label': str(training_labels.iloc[idx]),
            'cluster': int(kmeans.labels[idx]),
            'phrase': str(training_phrases.iloc[idx])
        }
        nearest_instances.append(instance_info)

    return nearest_instances


## PREDECIR CLUSTER Y ENCONTRAR VECINOS PARA NUEVAS INSTANCIAS ##
def predict_with_neighbors(kmeans_model, new_instance, new_phrase, training_data, training_labels, training_phrases,
                           n_neighbors=5):
    """
    Predice el cluster para una nueva instancia y encuentra sus vecinos más cercanos
    """
    # Predecir cluster
    cluster_prediction = kmeans_model.predict([new_instance])[0]

    # Encontrar instancias más cercanas
    nearest_instances = find_nearest_instances(
        new_instance, training_data, training_labels, training_phrases, n_neighbors
    )

    return {
        'predicted_cluster': int(cluster_prediction),
        'new_phrase': new_phrase,
        'nearest_neighbors': nearest_instances
    }


## EJEMPLO DE USO CON INSTANCIAS DE TEST ##
print("\n🔮 Prediciendo para instancias de prueba...")

# Probar con algunas instancias de test
for i in range(min(3, len(X_test_clean))):
    print(f"\n--- Instancia de Test {i + 1} ---")

    result = predict_with_neighbors(
        kmeans,
        X_test_clean[i],
        test_phrases.iloc[i],  # Frase de la instancia de test
        X_train_clean,
        y_train_clean,
        train_phrases,
        n_neighbors=5
    )

    print(f"📊 Cluster predicho: {result['predicted_cluster']}")
    print(f"🏷️  Label real: {y_test_clean.iloc[i]}")
    print(f"💬 Frase analizada: \"{result['new_phrase']}\"")
    print("👥 5 instancias más cercanas en entrenamiento:")

    for j, neighbor in enumerate(result['nearest_neighbors']):
        print(f"   {j + 1}. Distancia: {neighbor['distance']:.4f}, "
              f"Cluster: {neighbor['cluster']}, Label: {neighbor['label']}")
        print(f"      Frase: \"{neighbor['phrase']}\"")


## FUNCIÓN PARA CARGAR EL MODELO Y HACER PREDICCIONES ##
def load_kmeans_model(filename, training_data, training_labels, training_phrases):
    """
    Carga un modelo K-Means desde JSON y lo prepara para predicciones
    """
    with open(filename, 'r') as f:
        model_data = json.load(f)

    # Crear una nueva instancia de CustomKMeans
    loaded_model = CustomKMeans(
        n_clusters=model_data['n_clusters'],
        random_state=42,
        distance_metric=model_data.get('distance_metric', 'cosine')
    )

    # Configurar los atributos necesarios
    loaded_model.centroids = np.array(model_data['centroids'])
    loaded_model.inertia_ = model_data['inertia']
    loaded_model.n_iter_ = model_data['n_iter']

    return loaded_model, training_data, training_labels, training_phrases


# Ejemplo de cómo cargar y usar el modelo guardado
print("\n🔄 Cargando modelo guardado...")
loaded_kmeans, loaded_training_data, loaded_training_labels, loaded_training_phrases = load_kmeans_model(
    'kmeans_model.json', X_train_clean, y_train_clean, train_phrases
)

# Verificar que funciona
test_instance_idx = 0
result_loaded = predict_with_neighbors(
    loaded_kmeans,
    X_test_clean[test_instance_idx],
    test_phrases.iloc[test_instance_idx],
    loaded_training_data,
    loaded_training_labels,
    loaded_training_phrases
)

print(f"✅ Modelo cargado correctamente")
print(f"📊 Cluster predicho con modelo cargado: {result_loaded['predicted_cluster']}")
print(f"💬 Frase analizada: \"{result_loaded['new_phrase']}\"")

## ANÁLISIS DE RESULTADOS ##
print("\n📈 ANÁLISIS DE RESULTADOS:")
print(f"K utilizado: {chosen_k}")
print(f"Métrica de distancia: {kmeans.distance_metric}")
print(f"Inercia final: {kmeans.inertia_:.2f}")
print(f"Iteraciones: {kmeans.n_iter_}")