import torch
from torch import Tensor, nn

SHRINK = 5

class Recommender(nn.Module):
    def __init__(self, game_features: Tensor, creators: Tensor, n_creator_ids: int, mean_rating: float, *,
                 use_ids: bool, use_content: bool, use_creators: bool, embedding_size: int = 32, hidden_sizes: tuple[int, ...] = ()):
        super().__init__()
        self.register_buffer("game_features", game_features)
        self.register_buffer("creators", creators)
        self.n_creator_ids = n_creator_ids
        self.register_buffer("mean_rating", torch.tensor(mean_rating))
        self.use_ids = use_ids
        self.use_content = use_content
        self.use_creators = use_creators
        self.hidden_sizes = hidden_sizes
        if use_ids:
            self.game_embeddings = nn.Embedding(game_features.shape[0], embedding_size)
            nn.init.normal_(self.game_embeddings.weight, std=0.1)
        if use_content:
            self.game_content = nn.Linear(self.game_features.shape[1], embedding_size)
            nn.init.normal_(self.game_content.weight, std=0.1)
        if use_creators:
            self.game_creators = nn.Embedding(n_creator_ids, embedding_size)
            nn.init.normal_(self.game_creators.weight, std=0.1)
            self.register_buffer("creator_mask", self.creators != 0)
        if hidden_sizes:
            layers = []
            size_in = 3 * embedding_size
            for size in hidden_sizes:
                layers += [nn.Linear(size_in, size), nn.ReLU()]
                size_in = size
            layers.append(nn.Linear(size_in, 1))
            self.head = nn.Sequential(*layers)
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        self.game_offset = nn.Embedding(game_features.shape[0], 1)
        nn.init.zeros_(self.game_offset.weight)
    
    def _for_games(self, items: Tensor) -> Tensor:
        return self.mean_rating + self.game_offset(items).squeeze(-1)
    
    def _game_vectors(self, items: Tensor) -> Tensor:
        total = 0
        if self.use_ids:
            total = total + self.game_embeddings(items)
        if self.use_content:
            total = total + self.game_content(self.game_features[items])
        if self.use_creators:
            creator_vectors = self.game_creators(self.creators) * self.creator_mask[:, :, None]
            per_game = creator_vectors.sum(1) / self.creator_mask.sum(1)[:, None].clamp(min=1)
            total = total + per_game[items]
        return total
    
    def forward(self,
                history_items: Tensor,
                history_ratings: Tensor,
                history_mask: Tensor,
                target_items: Tensor) -> Tensor:
        
        history_games = self._game_vectors(history_items)
        centered_ratings = (history_ratings - self._for_games(history_items)) * history_mask
        user_offset = centered_ratings.sum(1) / (history_mask.sum(1) + SHRINK)
        centered_games = history_games * centered_ratings[:, :, None]
        user_games = centered_games.sum(1) / history_mask.sum(1)[:, None].clamp(min=1)
        target_games = self._game_vectors(target_items)
        pred_games = (user_games[:, None, :] * target_games).sum(-1)
        pred_ratings = self._for_games(target_items) + pred_games + user_offset[:, None]
        if self.hidden_sizes:
            user_per_target = user_games[:, None, :].expand(-1, target_items.shape[1], -1)
            hidden_targets = user_per_target * target_games
            glue_games = torch.cat([user_per_target, target_games, hidden_targets], dim=-1)
            net_games = self.head(glue_games).squeeze(-1)
            pred_ratings = pred_ratings + net_games
        return pred_ratings