import torch
from torch import Tensor, nn

GAME_EMBEDDING_SIZE = 32
SHRINK = 5

class Recommender(nn.Module):
    def __init__(self, game_features: Tensor, creators: Tensor, n_creator_ids: int, mean_rating: float, *,
                 use_ids: bool, use_content: bool, use_creators: bool):
        super().__init__()
        self.register_buffer("game_features", game_features)
        self.register_buffer("creators", creators)
        self.n_creator_ids = n_creator_ids
        self.register_buffer("mean_rating", torch.tensor(mean_rating))
        self.use_ids = use_ids
        self.use_content = use_content
        self.use_creators = use_creators
        self.game_embeddings = nn.Embedding(game_features.shape[0], GAME_EMBEDDING_SIZE)
        nn.init.normal_(self.game_embeddings.weight, std=0.1)
        self.game_offset = nn.Embedding(game_features.shape[0], 1)
        nn.init.zeros_(self.game_offset.weight)
    
    def _for_games(self, items: Tensor) -> Tensor:
        return self.mean_rating + self.game_offset(items).squeeze(-1)
    
    def forward(self,
                history_items: Tensor,
                history_ratings: Tensor,
                history_mask: Tensor,
                target_items: Tensor) -> Tensor:
        
        history_games = self.game_embeddings(history_items)
        centered_ratings = (history_ratings - self._for_games(history_items)) * history_mask
        user_offset = centered_ratings.sum(1) / (history_mask.sum(1) + SHRINK)
        centered_games = history_games * centered_ratings[:, :, None]
        user_games = centered_games.sum(1) / history_mask.sum(1)[:, None].clamp(min=1)
        target_games = self.game_embeddings(target_items)
        pred_games = (user_games[:, None, :] * target_games).sum(-1)
        pred_ratings = self._for_games(target_items) + pred_games + user_offset[:, None]
        return pred_ratings