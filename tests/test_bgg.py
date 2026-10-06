import polars as pl
import pytest

from bgg.data import load_ratings, split
from bgg.evaluate import average_rating_predictions, r_squared
from bgg.features import engineered


def test_r_squared_ignores_scale_but_not_order():
    assert r_squared([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert r_squared([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(1)
    assert r_squared([1, 2, 1, 2], [1, 1, 2, 2]) == pytest.approx(0)


def test_average_rating_uses_only_training_ratings():
    train = pl.DataFrame({"item_id": [1, 1, 2], "user_id": ["a", "b", "a"], "rating": [6.0, 8.0, 3.0]})
    test = pl.DataFrame({"item_id": [1, 2, 3], "user_id": ["c", "c", "c"], "rating": [10.0, 10.0, 10.0]})
    scored = average_rating_predictions(train, test).sort("item_id")
    # game 3 has no training ratings, so it can't be predicted
    assert scored["item_id"].to_list() == [1, 2]
    assert scored["predicted"].to_list() == [7.0, 3.0]


def test_load_ratings_keeps_the_user_named_null_as_the_notebooks_did(tmp_path):
    (tmp_path / "user_ratings.csv").write_text("BGGId,Rating,Username\n1,8.0,alice\n2,5.0,null\n")
    ratings = load_ratings(tmp_path)
    assert ratings.columns == ["item_id", "user_id", "rating"]
    assert ratings["user_id"].to_list() == ["alice", ""]


def test_split_holds_users_out_of_training():
    # 50 users with 10 ratings each, and one user with a single rating
    ratings = pl.DataFrame(
        {
            "item_id": list(range(10)) * 50 + [0],
            "user_id": [f"u{u}" for u in range(50) for _ in range(10)] + ["once"],
            "rating": [float(i % 10) for i in range(500)] + [5.0],
        }
    )
    train, query, test = split(ratings)
    train_users, query_users, test_users = (set(part["user_id"]) for part in (train, query, test))
    assert not train_users & test_users
    assert query_users == test_users
    assert "once" not in train_users | test_users
    assert len(train) + len(query) + len(test) == 500
    # each held-out user's ratings are divided 7 to 3
    assert test.group_by("user_id").len()["len"].unique().to_list() == [3]


def test_engineered_player_count_features():
    games = pl.DataFrame(
        {
            "GoodPlayers": ["['3', '4']", "[]", "['8+']", "['2']"],
            "MinPlayers": [2, 1, 4, 2],
            "MaxPlayers": [5, 4, 10, 0],
            "NumUserRatings": [10, 10, 10, 10],
            "NumOwned": [20, 5, 20, 20],
            "MfgAgeRec": [12, 8, 10, 10],
            "ComAgeRec": [10.0, None, 10.0, 10.0],
            "MfgPlaytime": [60, 30, 45, 45],
            "ComMinPlaytime": [30, 30, 60, 45],
            "ComMaxPlaytime": [60, 30, 90, 45],
            "NumWish": [4, 0, 0, 0],
            "NumWant": [6, 0, 0, 0],
            "NumWeightVotes": [5, 0, 0, 0],
            "GameWeight": [3.0, 2.0, 0.0, 0.0],
        }
    )
    out = engineered(games)
    assert out["RatioScarcityHype"][0] == 0.5  # 10 people want it, 20 own it
    assert out["RatioWeightVotes"][0] == 0.5
    assert out["RatioWeightPerHour"][0] == pytest.approx(3.0 / 60)
    assert out["PlayerCountRatio"].to_list() == [0.5, 0, pytest.approx(1 / 7), 0]
    assert out["GoodAt4Players"].to_list() == [True, False, False, False]
    assert out["GoodAt8Players"].to_list() == [False, False, True, False]
    assert out["OverHyped"].to_list() == [False, True, False, False]
    assert out["AgeRecDiff"].to_list() == [2.0, None, 0.0, 0.0]
    assert out["MfgTimeAccuracy"].to_list() == [1, 0, -1, 0]


def test_engineered_ratios_survive_a_zero_divisor():
    games = pl.DataFrame(
        {
            "GoodPlayers": ["[]"], "MinPlayers": [1], "MaxPlayers": [1], "NumUserRatings": [0], "NumOwned": [0],
            "MfgAgeRec": [0], "ComAgeRec": [0.0], "MfgPlaytime": [0], "ComMinPlaytime": [0], "ComMaxPlaytime": [0],
            "NumWish": [3], "NumWant": [3], "NumWeightVotes": [0], "GameWeight": [0.0],
        }
    )
    out = engineered(games)
    for name in ("RatioScarcityHype", "RatioWeightVotes", "RatioWeightPerHour"):
        value = out[name][0]
        assert value is None or value == value and abs(value) != float("inf"), name
