"""Builds the table of game features the model looks up by game.

Run as `python -m bgg.features --data-dir data` to write, into the data directory:

- game_features.parquet: one row per game with item_id and float columns, no nulls
- game_creators.parquet: one row per game with item_id and the list of its
  creator ids (designers, artists, and publishers share one numbering, from 1)
- creators.parquet: what each creator id stands for
"""

import argparse
from pathlib import Path

import polars as pl
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor

# Straight from games.csv. Left out on purpose: AvgRating, BayesAvgRating,
# StdDev and the Rank columns, which are computed from every user's ratings,
# including the held-out users the model is tested on.
GAME_COLUMNS = [
    "GameWeight", "YearPublished", "MinPlayers", "MaxPlayers", "BestPlayers",
    "ComAgeRec", "MfgAgeRec", "LanguageEase", "MfgPlaytime", "ComMinPlaytime",
    "ComMaxPlaytime", "NumOwned", "NumWant", "NumWish", "NumUserRatings",
    "NumAlternates", "NumExpansions", "NumImplementations", "IsReimplementation",
    "Kickstarted", "Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family",
    "Cat:CGS", "Cat:Abstract", "Cat:Party", "Cat:Childrens",
]  # fmt: skip

# GoodAt8Players means eight or more
PLAYER_COUNTS = range(1, 9)


def engineered(games: pl.DataFrame) -> pl.DataFrame:
    """The game-level features designed in the original content-based notebook."""
    # "['2', '3', '4']" -> [2, 3, 4]
    good = pl.col("GoodPlayers").str.extract_all(r"\d+").list.eval(pl.element().cast(pl.Int64))
    community_playtime = (pl.col("ComMinPlaytime") + pl.col("ComMaxPlaytime")) / 2
    seats = pl.col("MaxPlayers") - pl.col("MinPlayers") + 1
    return games.select(
        # more ratings than owners: people voting on a game they haven't played
        OverHyped=pl.col("NumUserRatings") > pl.col("NumOwned"),
        # how far the box's age recommendation is above the community's
        AgeRecDiff=pl.col("MfgAgeRec") - pl.col("ComAgeRec"),
        # 1 if the box promises a longer game than people report, -1 if shorter
        MfgTimeAccuracy=(pl.col("MfgPlaytime") - community_playtime).sign(),
        # share of the supported player counts the community calls good
        PlayerCountRatio=pl.when(seats > 0).then(good.list.len() / seats).otherwise(0),
        **{
            f"GoodAt{n}Players": good.list.eval(
                pl.element() >= n if n == PLAYER_COUNTS[-1] else pl.element() == n
            ).list.any()
            for n in PLAYER_COUNTS
        },
        RatioScarcityHype=pl.when(pl.col('NumOwned') > 0).then((pl.col('NumWish')+pl.col('NumWant'))/pl.col('NumOwned')).otherwise(0),
        RatioWeightVotes=pl.when(pl.col('NumUserRatings') > 0).then(pl.col('NumWeightVotes') / pl.col('NumUserRatings')).otherwise(0),
        RatioWeightPerHour=pl.when((pl.col('MfgPlaytime') > 0) & (pl.col('GameWeight') > 0)).then(pl.col('GameWeight') / pl.col('MfgPlaytime')).otherwise(None)
    )


def build_game_features(data_dir: str | Path) -> pl.DataFrame:
    data_dir = Path(data_dir)
    games = pl.read_csv(data_dir / "games.csv")
    flags = [
        pl.read_csv(data_dir / f"{name}.csv")
        for name in ("mechanics", "themes", "subcategories")
    ]
    mechanics = flags[0].drop("BGGId")
    features = (
        games.select(pl.col("BGGId").alias("item_id"), *GAME_COLUMNS)
        .hstack(engineered(games))
        .hstack(description_vectors(games))
        .hstack(mechanics.select(NumMechanics=pl.sum_horizontal(pl.all())))
    )
    for table in flags:
        features = features.join(table, left_on="item_id", right_on="BGGId", how="left")
    # a missing value (unknown age recommendation, say) becomes the column's median
    values = pl.exclude("item_id").cast(pl.Float32)
    return features.with_columns(values.fill_null(values.median()))

def description_vectors(games: pl.DataFrame) -> pl.DataFrame:
    vectorizer = TfidfVectorizer(min_df=5, max_features=20000)
    desc_vectors = vectorizer.fit_transform(games['Description'].fill_null('').to_list())
    svd = TruncatedSVD(n_components=64, random_state=0)
    embedded_desc = svd.fit_transform(desc_vectors)
    return pl.DataFrame(embedded_desc, schema=[f'Desc{i}' for i in range(64)], orient='row')

# long-tailed counts and durations, compressed with log(1 + x) before scaling
LOG_COLUMNS = [
    "NumOwned", "NumWant", "NumWish", "NumUserRatings", "NumAlternates",
    "NumExpansions", "NumImplementations", "MfgPlaytime", "ComMinPlaytime",
    "ComMaxPlaytime", "RatioScarcityHype", "RatioWeightPerHour"
]  # fmt: skip


RATIO_COLUMNS = ("RatioScarcityHype", "RatioWeightVotes", "RatioWeightPerHour")


def load_game_tensors(
    data_dir: str | Path, drop: tuple[str, ...] = (), max_creators: int = 32,
    descriptions: bool = False, ratios: bool = False,
) -> tuple[pl.Series, Tensor, list[str], Tensor, int]:
    """Reads the tables written by this module as tensors for a model.

    Returns (item ids, features, feature names, creators, number of creator ids).
    `features` is [games, features] with every non-flag column scaled to mean 0
    and standard deviation 1. `creators` is [games, max_creators] creator ids,
    padded with 0. Columns whose name starts with anything in `drop` are left
    out, which is how a run without the player-count features is made. The
    description and ratio columns are left out unless `descriptions` or
    `ratios` is set, so models trained before they existed still load.
    """
    data_dir = Path(data_dir)
    table = pl.read_parquet(data_dir / "game_features.parquet")
    if not descriptions:
        drop = (*drop, "Desc")
    # by exact name: one of the mechanics is called "Ratio / Combat Results Table"
    left_out = () if ratios else RATIO_COLUMNS
    names = [c for c in table.columns if c != "item_id" and not c.startswith(drop) and c not in left_out]
    scaled = []
    for name in names:
        column = table[name]
        if name == "YearPublished":
            # 0 means unknown, and a few ancient games are dated in the BCs
            column = column.clip(1950, None)
        if name in RATIO_COLUMNS:
            # a handful of games have ratios dozens of times the usual; cap them
            column = column.clip(upper_bound=column.quantile(0.99))
        if name in LOG_COLUMNS:
            column = column.log1p()
        if column.n_unique() > 2 and column.std() > 0:
            column = (column - column.mean()) / column.std()
        scaled.append(column.alias(name))
    features = pl.DataFrame(scaled).to_torch(dtype=pl.Float32)

    lists = pl.read_parquet(data_dir / "game_creators.parquet")["creators"]
    creators = torch.zeros(len(lists), max_creators, dtype=torch.long)
    for row, ids in enumerate(lists.list.head(max_creators).to_list()):
        creators[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    n_creator_ids = pl.read_parquet(data_dir / "creators.parquet").height + 1
    return table["item_id"], features, names, creators, n_creator_ids


def load_family_ids(data_dir: str | Path, item_ids: pl.Series) -> tuple[Tensor, int]:
    """Each game's family id in the order of `item_ids`, and how many ids there are counting 0 for "no family"."""
    data_dir = Path(data_dir)
    per_game = pl.DataFrame({"item_id": item_ids}).join(
        pl.read_parquet(data_dir / "game_families.parquet"), on="item_id", how="left", maintain_order="left"
    )
    return per_game["family_id"].fill_null(0).to_torch(), pl.read_parquet(data_dir / "families.parquet").height + 1


def build_game_creators(data_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (creators per game, what each creator id stands for).

    The source files have a 0/1 column per designer, artist, or publisher, with
    the rare ones already folded into one "Low-Exp" column per kind. A game with
    165 artists would be 165 ones in a row of 1,681 columns; a list of ids says
    the same thing and is what an embedding layer wants.
    """
    data_dir = Path(data_dir)
    credits = pl.concat(
        pl.read_csv(data_dir / f"{kind}s_reduced.csv")
        .unpivot(index="BGGId", variable_name="name")
        .filter(pl.col("value") == 1)
        .select(item_id="BGGId", kind=pl.lit(kind), name="name")
        for kind in ("designer", "artist", "publisher")
    )
    vocabulary = (
        credits.select("kind", "name").unique().sort("kind", "name")
        # id 0 is left free to mean "no creator" when lists are padded
        .with_row_index("creator_id", offset=1)
    )
    per_game = (
        credits.join(vocabulary, on=["kind", "name"])
        .group_by("item_id")
        .agg(creators=pl.col("creator_id").sort())
    )
    games = pl.read_csv(data_dir / "games.csv", columns=["BGGId"]).select(item_id="BGGId")
    per_game = games.join(per_game, on="item_id", how="left").with_columns(
        pl.col("creators").fill_null([])
    )
    return per_game, vocabulary

def build_game_families(data_dir: str | Path, min_games: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    data_dir = Path(data_dir)
    games = pl.read_csv(data_dir / 'games.csv', columns=['BGGId', 'Family'])
    counts = games.drop_nulls('Family').group_by('Family').len()
    counts = counts.filter(pl.col('len') >= min_games).sort('Family')
    vocabulary = counts.with_row_index('family_id', offset=1)
    per_game = games.join(vocabulary, on='Family', how='left').with_columns(pl.col('family_id').cast(pl.Int64).fill_null(0)).select(item_id='BGGId', family_id='family_id')
    return per_game, vocabulary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    features = build_game_features(args.data_dir)
    out = Path(args.data_dir) / "game_features.parquet"
    features.write_parquet(out)
    print(f"wrote {out}: {features.height:,} games, {features.width - 1} features")

    per_game, vocabulary = build_game_creators(args.data_dir)
    per_game.write_parquet(Path(args.data_dir) / "game_creators.parquet")
    vocabulary.write_parquet(Path(args.data_dir) / "creators.parquet")
    print(f"wrote game_creators.parquet and creators.parquet: {vocabulary.height:,} creators")

    per_game, vocabulary = build_game_families(args.data_dir)
    per_game.write_parquet(Path(args.data_dir) / "game_families.parquet")
    vocabulary.write_parquet(Path(args.data_dir) / "families.parquet")
    print(f'wrote game_families.parquet and families.parquet: {vocabulary.height:,} families')