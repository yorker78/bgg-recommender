"""Loading the BoardGameGeek ratings and splitting them for evaluation."""

from pathlib import Path

import polars as pl
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_ratings(data_dir: str | Path) -> pl.DataFrame:
    """One row per rating: item_id, user_id, rating."""
    return pl.read_csv(Path(data_dir) / "user_ratings.csv").select(
        item_id="BGGId",
        # One user is named "null". The earlier notebooks' copy of this data
        # stored that name as an empty string, and the split depends on the
        # sort order of user ids, so it has to match to hold out the same users.
        user_id=pl.col("Username").replace("null", ""),
        rating="Rating",
    )


def query_test_split(
    held_out: pl.DataFrame, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divides each user's ratings 70/30 into a query set and a test set."""
    query, test = train_test_split(
        held_out, train_size=0.7, stratify=held_out["user_id"], random_state=seed
    )
    return query, test


def split(
    ratings: pl.DataFrame, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Returns (train, query, test).

    A fifth of the users are held out of training entirely. For each of them,
    70% of their ratings are the query (what the model is told about them) and
    30% are the test (what it has to predict). This is the split the kNN
    notebook in traditional-recommenders used, so results are comparable.
    """
    # a user needs at least two ratings to appear in both query and test
    ratings = ratings.filter(pl.col("user_id").is_duplicated())
    train_idx, held_out_idx = next(
        GroupShuffleSplit(n_splits=5, random_state=seed).split(
            ratings, groups=ratings["user_id"]
        )
    )
    query, test = query_test_split(ratings[held_out_idx], seed)
    return ratings[train_idx], query, test


def hold_out_validation_users(
    train: pl.DataFrame, share: float = 0.05, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Returns (train, validation query, validation test).

    Takes a few users out of training to decide when to stop, so the test
    users are only ever used for the final score.
    """
    users = train["user_id"].unique().sort().sample(fraction=share, seed=seed)
    is_validation = pl.col("user_id").is_in(users.implode())
    query, test = query_test_split(train.filter(is_validation), seed)
    return train.filter(~is_validation), query, test
