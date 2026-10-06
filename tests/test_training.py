"""Checks on the batching, scoring, and training loop, using made-up data
small enough that the right answer is known."""

import polars as pl
import pytest
import torch

from bgg.baselines import BiasModel
from bgg.batches import evaluation_batches, game_index, to_table, training_batches
from bgg.evaluate import average_rating_predictions, r_squared, score_model


def toy_ratings() -> pl.DataFrame:
    """40 users and 20 games. Half the users love even-numbered games (9) and
    dislike odd ones (3); the other half are the reverse. Every game's average
    is 6, so nothing that ignores the user can tell the ratings apart."""
    rows = [
        {"user_id": f"u{user:02}", "item_id": 100 + game, "rating": 9.0 if (user + game) % 2 == 0 else 3.0}
        for user in range(40)
        for game in range(20)
    ]
    # shuffled, so nothing can rely on a user's ratings arriving together
    return pl.DataFrame(rows).sample(fraction=1, shuffle=True, seed=1)


@pytest.fixture
def toy():
    ratings = toy_ratings()
    index = game_index(ratings["item_id"].unique().sort())
    users = ratings["user_id"].unique().sort()
    return ratings, index, users


def test_table_groups_each_users_ratings(toy):
    ratings, index, users = toy
    table = to_table(ratings, index, users)
    assert table.counts.tolist() == [20] * 40
    for user in (0, 7, 39):
        rows = slice(int(table.offsets[user]), int(table.offsets[user] + table.counts[user]))
        expected = ratings.filter(pl.col("user_id") == users[user]).sort("item_id")
        got = sorted(zip(table.items[rows].tolist(), table.ratings[rows].tolist()))
        assert got == [(item - 100, rating) for item, rating in zip(expected["item_id"], expected["rating"])]


def test_training_batches_never_show_a_target_in_the_history(toy):
    ratings, index, users = toy
    table = to_table(ratings, index, users)
    seen_users = 0
    for batch in training_batches(table, 16, 8, 4, torch.Generator().manual_seed(0)):
        for row in range(len(batch.history_items)):
            history = set(batch.history_items[row][batch.history_mask[row]].tolist())
            targets = set(batch.target_items[row][batch.target_mask[row]].tolist())
            # each user rated each game once, so any overlap would be a leak
            assert not history & targets
            assert len(history) == 8 and len(targets) == 4
            # the ratings came along with the right games: this user's pattern
            parity = {(game + int(rating == 9.0)) % 2 for game, rating in zip(
                batch.history_items[row][batch.history_mask[row]].tolist(),
                batch.history_ratings[row][batch.history_mask[row]].tolist(),
            )}
            assert len(parity) == 1
            seen_users += 1
    assert seen_users == 40


def test_evaluation_batches_use_query_as_history_and_test_as_targets(toy):
    ratings, index, users = toy
    query = to_table(ratings.filter(pl.col("item_id") < 110), index, users)
    # user u00 has no test ratings at all, and must still line up
    test = to_table(ratings.filter((pl.col("item_id") >= 110) & (pl.col("user_id") != "u00")), index, users)
    batches = list(evaluation_batches(query, test, users_per_batch=16, max_history=1024))
    assert sum(len(b.target_items) for b in batches) == 40
    first = batches[0]
    assert first.target_mask[0].sum() == 0 and first.target_mask[1].sum() == 10
    assert set(first.history_items[1][first.history_mask[1]].tolist()) == set(range(10))
    assert set(first.target_items[1][first.target_mask[1]].tolist()) == set(range(10, 20))


def test_score_model_matches_the_average_rating_baseline():
    # games with different averages, so the baseline has something to predict
    ratings = pl.DataFrame(
        {
            "user_id": [f"u{u}" for u in range(30) for _ in range(6)],
            "item_id": [g for _ in range(30) for g in range(6)],
            "rating": [float((g * 3 + u) % 10) for u in range(30) for g in range(6)],
        }
    )
    train = ratings.filter(pl.col("user_id") < "u2")
    held_out = ratings.filter(pl.col("user_id") >= "u2")
    query, test = held_out.filter(pl.col("item_id") < 3), held_out.filter(pl.col("item_id") >= 3)
    index = game_index(ratings["item_id"].unique().sort())
    users = test["user_id"].unique().sort()

    means = train.group_by("item_id").agg(pl.col("rating").mean()).sort("item_id")
    model = BiasModel(6, 0.0)
    model.game_offset.weight.data = means["rating"].to_torch().float()[:, None]
    scores = score_model(model, to_table(query, index, users), to_table(test, index, users))

    baseline = average_rating_predictions(train, test)
    assert scores["ratings"] == len(test)
    assert scores["r2"] == pytest.approx(r_squared(baseline["predicted"], baseline["rating"]))
    assert scores["mae"] == pytest.approx((baseline["predicted"] - baseline["rating"]).abs().mean())


def models_to_check():
    """Every model has to pass the same checks. The real one joins when it exists."""
    yield "game averages", lambda: BiasModel(20, 6.0)
    yield "game averages with user offset", lambda: BiasModel(20, 6.0, user_offset=True)
    try:
        from bgg.model import Recommender
    except ImportError:
        return
    features = torch.zeros(20, 3)
    creators = torch.zeros(20, 4, dtype=torch.long)
    yield "recommender", lambda: Recommender(features, creators, 1, 6.0, use_ids=True, use_content=False, use_creators=False)
    described = torch.randn(20, 3)
    yield "recommender with content", lambda: Recommender(described, creators, 1, 6.0, use_ids=True, use_content=True, use_creators=False)
    yield "recommender, content only", lambda: Recommender(described, creators, 1, 6.0, use_ids=False, use_content=True, use_creators=False)
    credited = torch.randint(0, 5, (20, 4))  # 0 is "no creator"
    yield "recommender with creators", lambda: Recommender(described, credited, 5, 6.0, use_ids=True, use_content=True, use_creators=True)


@pytest.mark.parametrize("name,make", list(models_to_check()))
def test_model_output_shape_and_empty_histories(toy, name, make):
    ratings, index, users = toy
    model = make()
    history_items = torch.zeros(5, 7, dtype=torch.long)
    history_ratings = torch.zeros(5, 7)
    no_history = torch.zeros(5, 7, dtype=torch.bool)
    targets = torch.arange(15).reshape(5, 3)
    output = model(history_items, history_ratings, no_history, targets)
    assert output.shape == (5, 3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("name,make", list(models_to_check()))
def test_padding_does_not_change_predictions(toy, name, make):
    """Whatever sits in the masked-out part of a history must be ignored."""
    model = make().eval()
    items = torch.randint(0, 20, (4, 6))
    ratings = torch.rand(4, 6) * 10
    mask = torch.tensor([[True, True, True, False, False, False]] * 4)
    targets = torch.randint(0, 20, (4, 5))
    with torch.no_grad():
        before = model(items, ratings, mask, targets)
        items[:, 3:], ratings[:, 3:] = 19, 10.0
        after = model(items, ratings, mask, targets)
    assert torch.allclose(before, after)


def learns_the_toy_pattern(model, ratings, index, users, epochs=200, lr=0.05) -> float:
    table = to_table(ratings.filter(pl.col("user_id") < "u30"), index, users.filter(users < "u30"))
    held_out = ratings.filter(pl.col("user_id") >= "u30")
    held_out_users = held_out["user_id"].unique().sort()
    query = to_table(held_out.filter(pl.col("item_id") < 114), index, held_out_users)
    test = to_table(held_out.filter(pl.col("item_id") >= 114), index, held_out_users)
    generator = torch.Generator().manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for batch in training_batches(table, 30, 14, 6, generator):
            predicted = model(batch.history_items, batch.history_ratings, batch.history_mask, batch.target_items)
            loss = ((predicted - batch.target_ratings)[batch.target_mask] ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return score_model(model, query, test)["mae"]


def test_game_averages_cannot_learn_the_toy_pattern(toy):
    """The toy data is built so that ignoring the user gets you nowhere. If
    this ever passes with a low error, the data or the scoring is leaking."""
    torch.manual_seed(0)
    assert learns_the_toy_pattern(BiasModel(20, 6.0), *toy) > 2.5


def test_recommender_learns_the_toy_pattern(toy):
    """Users unseen in training, whose taste is obvious from their history."""
    model_module = pytest.importorskip("bgg.model", reason="bgg/model.py isn't written yet")
    torch.manual_seed(0)
    model = model_module.Recommender(
        torch.zeros(20, 3), torch.zeros(20, 4, dtype=torch.long), 1, 6.0,
        use_ids=True, use_content=False, use_creators=False,
    )
    assert learns_the_toy_pattern(model, *toy) < 0.5
