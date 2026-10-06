import torch
from torch import Tensor, nn

class TwoTower(nn.Module):
    def __init__(self,
                 game_features: Tensor,
                 mean_rating: float,
                 *,
                 embedding_size: int = 32,
                 hidden_size: int = 64,
                 cosine: bool =False):
        super().__init__()
        self.register_buffer("game_features", game_features)
        self.register_buffer("mean_rating", torch.tensor(mean_rating))
        self.game_tower = nn.Linear(game_features.shape[1], embedding_size)
        nn.init.normal_(self.game_tower.weight, std=0.1)
        self.user_tower = nn.Sequential(nn.Linear(2 * embedding_size + 1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, embedding_size))
        self.cosine = cosine
        if cosine:
            self.temperature = nn.Parameter(torch.tensor(10.0))
        
    def forward(self,
                history_items: Tensor,
                history_ratings: Tensor,
                history_mask: Tensor,
                target_items: Tensor) -> Tensor:
        ideal = self.user_vectors(history_items, history_ratings, history_mask)
        targets = self.game_vectors(target_items)
        if self.cosine:
            ideal = torch.nn.functional.normalize(ideal, dim=-1)
            targets = torch.nn.functional.normalize(targets, dim=-1)
        scores = (ideal[:, None, :] * targets).sum(-1)
        return scores * self.temperature if self.cosine else scores
    
    def game_vectors(self, items: Tensor) -> Tensor:
        return self.game_tower(self.game_features[items])
    
    def user_vectors(self,
                     history_items: Tensor,
                     history_ratings: Tensor,
                     history_mask: Tensor) -> Tensor:
        history_games = self.game_vectors(history_items)
        centered_ratings = (history_ratings - self.mean_rating) * history_mask
        centered_games = history_games * centered_ratings[:, :, None]
        user_games = centered_games.sum(1) / history_mask.sum(1)[:, None].clamp(min=1)
        rated_games = history_games * history_mask[:, :, None]
        seen_games = rated_games.sum(1) / (history_mask.sum(1)[:, None].clamp(min=1))
        history_length = torch.log1p(history_mask.sum(1).float())[:, None]
        return self.user_tower(torch.cat([user_games, seen_games, history_length], dim=-1))