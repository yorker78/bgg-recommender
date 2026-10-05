"""Builds the table of game features the model looks up by game.

Run as `python -m bgg.features --data-dir data` to write data/game_features.parquet,
one row per game with item_id and float columns, no nulls.
"""

import argparse
from pathlib import Path

import polars as pl

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
        .hstack(mechanics.select(NumMechanics=pl.sum_horizontal(pl.all())))
    )
    for table in flags:
        features = features.join(table, left_on="item_id", right_on="BGGId", how="left")
    # a missing value (unknown age recommendation, say) becomes the column's median
    values = pl.exclude("item_id").cast(pl.Float32)
    return features.with_columns(values.fill_null(values.median()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    features = build_game_features(args.data_dir)
    out = Path(args.data_dir) / "game_features.parquet"
    features.write_parquet(out)
    print(f"wrote {out}: {features.height:,} games, {features.width - 1} features")
