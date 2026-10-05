"""The simplest things that fit the model interface, as reference points."""

import torch
from torch import Tensor, nn


class BiasModel(nn.Module):
    """Predicts the overall mean rating plus a learned offset for each game.

    With `user_offset`, it also shifts every prediction for a user by how far
    their history's ratings sit above or below what it would have predicted
    for those games: someone who rates everything a point high gets a point
    added.
    """

    # ponytail: fixed shrinkage toward no offset for short histories; make it
    # learned if this model is ever more than a reference point
    SHRINK = 5.0

    def __init__(self, n_games: int, mean_rating: float, user_offset: bool = False):
        super().__init__()
        self.game_offset = nn.Embedding(n_games, 1)
        nn.init.zeros_(self.game_offset.weight)
        self.register_buffer("mean", torch.tensor(mean_rating))
        self.user_offset = user_offset

    def _for_games(self, items: Tensor) -> Tensor:
        return self.mean + self.game_offset(items).squeeze(-1)

    def forward(
        self,
        history_items: Tensor,
        history_ratings: Tensor,
        history_mask: Tensor,
        target_items: Tensor,
    ) -> Tensor:
        predicted = self._for_games(target_items)
        if self.user_offset:
            surprise = (history_ratings - self._for_games(history_items)) * history_mask
            offset = surprise.sum(1) / (history_mask.sum(1) + self.SHRINK)
            predicted = predicted + offset[:, None]
        return predicted
