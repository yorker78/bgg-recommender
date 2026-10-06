"""Scoring predictions, and the average-rating baseline every model has to beat.

Run as `python -m bgg.evaluate --data-dir data` to print the baseline's score.
"""

import argparse

import polars as pl
import torch
from scipy.stats import pearsonr

from bgg.batches import RatingTable, evaluation_batches
from bgg.data import load_ratings, split


def r_squared(predicted, actual) -> float:
    """Squared Pearson correlation between predicted and actual ratings.

    This is the measure the traditional-recommenders notebooks report. It is
    not scikit-learn's r2_score, which also penalizes predictions that are
    correlated with the truth but on the wrong scale.
    """
    return pearsonr(predicted, actual).statistic ** 2


def average_rating_predictions(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """Predicts each game's mean training rating for every test rating of it.

    Test ratings of games with no training ratings are dropped.
    """
    means = train.group_by("item_id").agg(predicted=pl.col("rating").mean())
    return test.join(means, on="item_id", how="inner")


@torch.no_grad()
def score_model(
    model: torch.nn.Module,
    query: RatingTable,
    test: RatingTable,
    # small, because a model with attention builds a [users, targets, history]
    # tensor and one user can have over a thousand of each
    users_per_batch: int = 64,
    max_history: int = 1024,
) -> dict[str, float]:
    """R2 and mean absolute error (in rating points) over every test rating."""
    model.eval()
    predicted, actual = [], []
    for batch in evaluation_batches(query, test, users_per_batch, max_history):
        output = model(batch.history_items, batch.history_ratings, batch.history_mask, batch.target_items)
        predicted.append(output[batch.target_mask].cpu())
        actual.append(batch.target_ratings[batch.target_mask].cpu())
    predicted, actual = torch.cat(predicted), torch.cat(actual)
    return {
        "r2": float(r_squared(predicted.numpy(), actual.numpy())),
        "mae": float((predicted - actual).abs().mean()),
        "ratings": len(actual),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    train, query, test = split(load_ratings(args.data_dir))
    print(f"train {len(train):,} | query {len(query):,} | test {len(test):,} ratings")
    scored = average_rating_predictions(train, test)
    print(f"average rating R2: {r_squared(scored['predicted'], scored['rating']):.4f}")
