"""Turning ratings into the batches a model trains and is scored on.

A model never gets a user id. Each row of a batch is one user, described by a
history of (game, rating) pairs, and the model predicts their ratings for a
separate list of target games. For held-out users the history is their query
ratings and the targets are their test ratings. Training imitates that by
hiding part of each training user's ratings every epoch.
"""

from dataclasses import dataclass
from typing import Iterator, NamedTuple

import polars as pl
import torch
from torch import Tensor


class Batch(NamedTuple):
    """Rows are users. Histories and targets are padded to the same length, and
    the masks say which entries are real."""

    history_items: Tensor  # [users, history] game indices
    history_ratings: Tensor  # [users, history]
    history_mask: Tensor  # [users, history] bool
    target_items: Tensor  # [users, targets] game indices
    target_ratings: Tensor  # [users, targets]
    target_mask: Tensor  # [users, targets] bool


@dataclass
class RatingTable:
    """Ratings grouped by user: user u's are at offsets[u] : offsets[u] + counts[u]."""

    items: Tensor  # game index of each rating
    ratings: Tensor
    offsets: Tensor
    counts: Tensor

    def to(self, device) -> "RatingTable":
        return RatingTable(*(t.to(device) for t in (self.items, self.ratings, self.offsets, self.counts)))


def game_index(item_ids: pl.Series) -> pl.DataFrame:
    """Maps each item_id to its row in the game feature table."""
    return pl.DataFrame({"item_id": item_ids}).with_row_index("item")


def to_table(ratings: pl.DataFrame, index: pl.DataFrame, users: pl.Series) -> RatingTable:
    """Groups `ratings` by user, in the order given by `users`.

    A user in `users` with no ratings here gets a count of 0, so two tables
    built with the same `users` (a query table and a test table) line up.
    """
    order = pl.DataFrame({"user_id": users}).with_row_index("user")
    rows = ratings.join(index, on="item_id").join(order, on="user_id").sort("user")
    counts = (
        order.join(rows.group_by("user").len(), on="user", how="left")
        .sort("user")["len"]
        .fill_null(0)
        .to_torch()
        .long()
    )
    return RatingTable(
        items=rows["item"].to_torch().long(),
        ratings=rows["rating"].to_torch().float(),
        offsets=counts.cumsum(0) - counts,
        counts=counts,
    )


def _gather(table: RatingTable, starts: Tensor, lengths: Tensor, width: int):
    """Reads up to `width` consecutive ratings per user, starting at `starts`."""
    positions = torch.arange(width, device=starts.device)
    mask = positions < lengths[:, None]
    # padded entries read position 0, which is harmless because they're masked
    rows = torch.where(mask, starts[:, None] + positions, 0)
    return table.items[rows], table.ratings[rows], mask


def training_batches(
    table: RatingTable,
    users_per_batch: int,
    max_history: int,
    max_targets: int,
    generator: torch.Generator,
    history_share: float = 0.7,
) -> Iterator[Batch]:
    """One pass over the training users, with each user's ratings freshly
    shuffled and divided into a history and targets, 70/30 like the held-out
    users. A user with more ratings than fit contributes a random sample.
    """
    device = table.items.device
    n_users = len(table.counts)
    # shuffle within each user: sort by (user, random number)
    user_of = torch.repeat_interleave(torch.arange(n_users, device=device), table.counts)
    noise = torch.rand(len(user_of), generator=generator, device=device)
    order = torch.argsort(user_of.double() + noise.double())
    shuffled = RatingTable(table.items[order], table.ratings[order], table.offsets, table.counts)

    for users in torch.randperm(n_users, generator=generator, device=device).split(users_per_batch):
        counts = shuffled.counts[users]
        users, counts = users[counts >= 2], counts[counts >= 2]
        if len(users) == 0:
            continue
        n_history = (counts * history_share).round().long().clamp(1, max_history)
        n_history = torch.minimum(n_history, counts - 1)
        n_targets = (counts - n_history).clamp(max=max_targets)
        starts = shuffled.offsets[users]
        history = _gather(shuffled, starts, n_history, int(n_history.max()))
        targets = _gather(shuffled, starts + n_history, n_targets, int(n_targets.max()))
        yield Batch(*history, *targets)


def evaluation_batches(
    query: RatingTable, test: RatingTable, users_per_batch: int, max_history: int
) -> Iterator[Batch]:
    """Batches for scoring: all of each user's test ratings as targets, and
    their query ratings (the first `max_history` of them) as the history.
    The two tables must have been built with the same `users`.
    """
    for users in torch.arange(len(test.counts), device=test.items.device).split(users_per_batch):
        n_history = query.counts[users].clamp(max=max_history)
        n_targets = test.counts[users]
        history = _gather(query, query.offsets[users], n_history, max(int(n_history.max()), 1))
        targets = _gather(test, test.offsets[users], n_targets, max(int(n_targets.max()), 1))
        yield Batch(*history, *targets)
