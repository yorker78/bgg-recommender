import numpy as np
import sklearn
import torch
from numpy import ndarray
from sklearn import cluster, metrics
from sklearn.metrics import silhouette_score
from torch import Tensor, nn

from bgg.two_tower import TwoTower


class MultiInterest(nn.Module):
    def __init__(self,
                 model: TwoTower,
                 method: str = "kmeans",
                 n_clusters: int = 2,
                 min_cluster_size: int = 5,
                 categories: Tensor | None = None):
        super().__init__()
        self.model = model
        self.method = method
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.register_buffer("categories", categories)
    
    def cluster(self, vectors: ndarray, items: Tensor) -> ndarray:
        if vectors.shape[0] < 2 * self.min_cluster_size:
            return np.zeros(vectors.shape[0], dtype=int)
        match self.method:
            case "kmeans":
                labels = sklearn.cluster.KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0).fit_predict(vectors)
            case "kmeans-auto":
                labels = np.zeros(vectors.shape[0], dtype=int)
                best = 0.1
                for k in (2, 3, 4):
                    candidate = sklearn.cluster.KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vectors)
                    score = silhouette_score(vectors, candidate)
                    if score > best:
                        best = score
                        labels = candidate
            case "dbscan":
                labels = sklearn.cluster.DBSCAN(eps=0.5, min_samples=self.min_cluster_size).fit_predict(vectors)
            case "hdbscan":
                labels = sklearn.cluster.HDBSCAN(min_cluster_size=self.min_cluster_size).fit_predict(vectors)
            case "agglomerative":
                labels = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=0.7, metric="cosine", linkage="average").fit_predict(vectors)
            case "categories":
                flags = self.categories[items]
                labels = flags.argmax(1)
                labels[flags.sum(1) == 0] = -1
                labels = labels.cpu().numpy()
            case _:
                raise Exception("undefined method")
        if (labels == -1).all():
            return np.zeros(vectors.shape[0], dtype=int)
        return labels
    
    @torch.no_grad()
    def forward(self,
                history_items: Tensor,
                history_ratings: Tensor,
                history_mask: Tensor,
                target_items: Tensor) -> Tensor:
        vectors = torch.nn.functional.normalize(self.model.game_vectors(history_items), dim=-1)
        labels = torch.full_like(history_items, -1)
        for u in range(len(history_items)):
            real = history_mask[u]
            found = self.cluster(vectors[u][real].cpu().numpy(), history_items[u][real])
            labels[u, real] = torch.from_numpy(found).to(labels)
        target_games = torch.nn.functional.normalize(self.model.game_vectors(target_items), dim=-1)
        best = torch.full(target_items.shape, float("-inf"), device=target_items.device)
        for c in range(int(labels.max()) + 1):
            in_cluster = history_mask & (labels == c)
            ideal = torch.nn.functional.normalize(self.model.user_vectors(history_items, history_ratings, in_cluster), dim=-1)
            scores = (ideal[:, None, :] * target_games).sum(-1)
            scores[~in_cluster.any(1)] = float("-inf")
            best = torch.max(best, scores)
        no_clusters = best.isinf().all(1)
        best[no_clusters] = self.model(history_items, history_ratings, history_mask, target_items)[no_clusters]
        return best