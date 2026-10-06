"""Checks on the batching, scoring, and training loop, using made-up data
small enough that the right answer is known."""

import polars as pl
import pytest
import torch

from bgg.baselines import BiasModel
from bgg.batches import evaluation_batches, game_index, to_table, training_batches
from bgg.evaluate import average_rating_predictions, r_squared, score_model
from bgg.retrieval import score_retrieval, taste_variety
from bgg.train import interest_overlap, ranking_loss


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
    yield "recommender with a head", lambda: Recommender(described, credited, 5, 6.0, use_ids=True, use_content=True, use_creators=True, hidden_sizes=(16, 8))
    yield "recommender with a head and seen", lambda: Recommender(described, credited, 5, 6.0, use_ids=True, use_content=True, use_creators=True, use_seen=True, hidden_sizes=(16,))
    yield "recommender with a head, seen, and context", lambda: Recommender(described, credited, 5, 6.0, use_ids=True, use_content=True, use_creators=True, use_seen=True, use_context=True, hidden_sizes=(16,))
    yield "recommender with attention and no head", lambda: Recommender(described, credited, 5, 6.0, use_ids=True, use_content=True, use_creators=True, use_attention=True)
    yield "recommender with everything on", lambda: Recommender(described, credited, 5, 6.0, use_ids=True, use_content=True, use_creators=True, use_seen=True, use_context=True, use_attention=True, hidden_sizes=(16,))
    try:
        from bgg.two_tower import TwoTower
    except ImportError:
        return
    yield "two tower", lambda: TwoTower(described, 6.0, embedding_size=8, hidden_size=16)
    yield "two tower, cosine", lambda: TwoTower(described, 6.0, embedding_size=8, hidden_size=16, cosine=True)
    try:
        from bgg.two_tower import MultiTower
    except ImportError:
        return
    yield "multi tower", lambda: MultiTower(described, 6.0, embedding_size=8, hidden_size=16, n_interests=3)


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


def test_score_model_keeps_predictions_on_the_rating_scale():
    """Predictions of -94, 6 and 106 are scored as 1, 6 and 10."""
    model = BiasModel(3, 6.0)
    model.game_offset.weight.data = torch.tensor([[-100.0], [0.0], [100.0]])
    index = game_index(pl.Series([0, 1, 2]))
    users = pl.Series(["a"])
    query = pl.DataFrame({"user_id": ["a"], "item_id": [0], "rating": [5.0]})
    test = pl.DataFrame({"user_id": ["a"] * 3, "item_id": [0, 1, 2], "rating": [1.0, 6.0, 10.0]})
    scores = score_model(model, to_table(query, index, users), to_table(test, index, users))
    assert scores["mae"] == 0


def test_retrieval_scores_a_known_ranking():
    """Six games ranked 5, 4, 3, 2, 1, 0 for everyone. The user's history is
    game 5, so their top recommendations are 4, 3, 2, 1, 0. They went on to
    like games 4 and 1 and to dislike game 3."""
    from bgg.baselines import Popularity

    index = game_index(pl.Series([0, 1, 2, 3, 4, 5]))
    users = pl.Series(["a", "b"])
    query = pl.DataFrame({"user_id": ["a", "b"], "item_id": [5, 5], "rating": [7.0, 7.0]})
    test = pl.DataFrame({
        "user_id": ["a", "a", "a", "b"],  # b likes nothing in test, so is not scored
        "item_id": [4, 1, 3, 4],
        "rating": [9.0, 8.0, 2.0, 5.0],
    })
    result = score_retrieval(
        Popularity(torch.arange(6)), to_table(query, index, users), to_table(test, index, users), n_games=6
    )
    assert result["users"] == 1
    assert result["hits@10"] == 2
    assert result["recall@50"] == 1.0
    # hits at ranks 1 and 4, against the best case of ranks 1 and 2
    log2 = torch.log2
    expected = (1 / log2(torch.tensor(2.0)) + 1 / log2(torch.tensor(5.0))) / (1 / log2(torch.tensor(2.0)) + 1 / log2(torch.tensor(3.0)))
    assert result["ndcg@10"] == pytest.approx(float(expected))

    # with only the 100 best-known games counted as well known, all six are, so nothing is beyond them
    counted = score_retrieval(
        Popularity(torch.arange(6)), to_table(query, index, users), to_table(test, index, users), n_games=6,
        rating_counts=torch.arange(6),
    )
    assert counted["beyond_hits@10"] == 0 and counted["well_known@10"] == 1.0
    assert counted["games_recommended"] == 5  # every game but the one in the history


def learns_to_rank_the_toy_pattern(model, ratings, index, users, steps=200, lr=0.05) -> float:
    """Trains on the ranking objective, then asks for held-out users' unseen
    games in order. Three of each user's six are ones they love."""
    table = to_table(ratings.filter(pl.col("user_id") < "u30"), index, users.filter(users < "u30"))
    held_out = ratings.filter(pl.col("user_id") >= "u30")
    held_out_users = held_out["user_id"].unique().sort()
    query = to_table(held_out.filter(pl.col("item_id") < 114), index, held_out_users)
    test = to_table(held_out.filter(pl.col("item_id") >= 114), index, held_out_users)
    generator = torch.Generator().manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        for batch in training_batches(table, 30, 14, 6, generator):
            loss = ranking_loss(model, batch, n_games=20, n_negatives=4, generator=generator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return score_retrieval(model, query, test, n_games=20)["ndcg@10"]


def test_recommender_learns_to_rank_the_toy_pattern(toy):
    model_module = pytest.importorskip("bgg.model")
    torch.manual_seed(0)
    model = model_module.Recommender(
        torch.zeros(20, 3), torch.zeros(20, 4, dtype=torch.long), 1, 6.0,
        use_ids=True, use_content=False, use_creators=False,
    )
    assert learns_to_rank_the_toy_pattern(model, *toy) > 0.95


@pytest.mark.parametrize("cosine", [False, True])
def test_two_tower_learns_to_rank_the_toy_pattern(toy, cosine):
    """It only sees content, so each toy game gets a feature of its own."""
    two_tower = pytest.importorskip("bgg.two_tower")
    torch.manual_seed(0)
    model = two_tower.TwoTower(torch.eye(20), 6.0, embedding_size=8, hidden_size=16, cosine=cosine)
    assert learns_to_rank_the_toy_pattern(model, *toy, lr=0.01) > 0.95


def test_taste_variety_and_scoring_a_chosen_group():
    """User a likes two war games; user b likes one war game and one party game."""
    index = game_index(pl.Series([0, 1, 2, 3]))
    users = pl.Series(["a", "b"])
    categories = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 0]])  # games 0 and 1 are war, 2 is party
    query = to_table(pl.DataFrame({
        "user_id": ["a", "a", "a", "b", "b"],
        "item_id": [0, 1, 2, 0, 2],
        "rating": [9.0, 8.0, 3.0, 9.0, 9.0],  # a's party game is disliked, so it doesn't count
    }), index, users)
    variety = taste_variety(query, categories, min_liked=2)
    assert taste_variety(query, categories, min_liked=3).isnan().all()  # neither has three liked games
    assert variety[0] == 0
    assert variety[1] == pytest.approx(float(torch.log(torch.tensor(2.0))))

    test = to_table(pl.DataFrame({"user_id": ["a", "b"], "item_id": [3, 3], "rating": [9.0, 9.0]}), index, users)
    from bgg.baselines import Popularity
    result = score_retrieval(Popularity(torch.arange(4)), query, test, n_games=4, only=variety > 0)
    assert result["users"] == 1


@pytest.mark.parametrize("method", ["kmeans", "kmeans-auto", "dbscan", "hdbscan", "agglomerative", "categories"])
def test_multi_interest_scores_like_the_model_it_wraps_when_it_cannot_split(method):
    """One ideal game per taste cluster. Users 0 and 1 have nothing to split
    (no history, and too short a history), so they must come out as the
    wrapped model would score them."""
    two_tower = pytest.importorskip("bgg.two_tower")
    multi_interest = pytest.importorskip("bgg.multi_interest")
    torch.manual_seed(0)
    n_games = 60
    categories = torch.zeros(n_games, 3, dtype=torch.long)
    categories[:20, 0], categories[20:40, 1] = 1, 1  # the last 20 games have no category
    model = two_tower.TwoTower(torch.randn(n_games, 6), 6.0, embedding_size=8, hidden_size=16, cosine=True).eval()
    wrapped = multi_interest.MultiInterest(model, method=method, categories=categories).eval()

    items = torch.stack([torch.randperm(n_games)[:30] for _ in range(4)])
    ratings = torch.rand(4, 30) * 10
    mask = torch.ones(4, 30, dtype=torch.bool)
    mask[0] = False
    mask[1, 6:] = False
    targets = torch.randint(0, n_games, (4, 12))
    with torch.no_grad():
        plain = model(items, ratings, mask, targets)
        scores = wrapped(items, ratings, mask, targets)
        items[1, 6:], ratings[1, 6:] = 0, 10.0  # padding must not matter
        again = wrapped(items, ratings, mask, targets)
    assert scores.shape == (4, 12) and torch.isfinite(scores).all()
    assert torch.allclose(scores, again, atol=1e-5)
    assert torch.allclose(scores[0], plain[0], atol=1e-5)
    # cosine similarity, which is the wrapped model's score before its temperature
    assert torch.allclose(scores[1], plain[1] / model.temperature, atol=1e-5)


def test_multi_interest_rejects_an_unknown_method():
    two_tower = pytest.importorskip("bgg.two_tower")
    multi_interest = pytest.importorskip("bgg.multi_interest")
    model = two_tower.TwoTower(torch.randn(30, 6), 6.0, embedding_size=8, hidden_size=16, cosine=True)
    wrapped = multi_interest.MultiInterest(model, method="kmean")
    with pytest.raises(Exception, match="undefined method"):
        wrapped(torch.arange(30)[None], torch.rand(1, 30) * 10, torch.ones(1, 30, dtype=torch.bool), torch.arange(5)[None])


def test_multi_tower_learns_to_rank_the_toy_pattern(toy):
    two_tower = pytest.importorskip("bgg.two_tower")
    if not hasattr(two_tower, "MultiTower"):
        pytest.skip("MultiTower isn't written yet")
    torch.manual_seed(0)
    model = two_tower.MultiTower(torch.eye(20), 6.0, embedding_size=8, hidden_size=16, n_interests=3)
    assert learns_to_rank_the_toy_pattern(model, *toy, lr=0.01) > 0.95
    # one ideal game per interest, for each user
    vectors = model.user_vectors(torch.zeros(4, 6, dtype=torch.long), torch.zeros(4, 6), torch.ones(4, 6, dtype=torch.bool))
    assert vectors.shape == (4, 3, 8)


def test_multi_tower_scores_a_target_by_its_best_matching_interest():
    two_tower = pytest.importorskip("bgg.two_tower")
    if not hasattr(two_tower, "MultiTower"):
        pytest.skip("MultiTower isn't written yet")
    torch.manual_seed(0)
    model = two_tower.MultiTower(torch.randn(20, 6), 6.0, embedding_size=8, hidden_size=16, n_interests=3).eval()
    history = (torch.randint(0, 20, (4, 7)), torch.rand(4, 7) * 10, torch.rand(4, 7) > 0.2)
    targets = torch.randint(0, 20, (4, 5))
    with torch.no_grad():
        scores = model(*history, targets)
        ideals = torch.nn.functional.normalize(model.user_vectors(*history), dim=-1)  # [users, interests, d]
        games = torch.nn.functional.normalize(model.game_vectors(targets), dim=-1)  # [users, targets, d]
    for user in range(4):
        for target in range(5):
            cosines = ideals[user] @ games[user, target]  # one per interest
            assert scores[user, target] == pytest.approx(float(cosines.max() * model.temperature), abs=1e-5)


def test_interest_overlap_is_one_when_every_interest_is_the_same():
    two_tower = pytest.importorskip("bgg.two_tower")
    if not hasattr(two_tower, "MultiTower"):
        pytest.skip("MultiTower isn't written yet")
    from bgg.batches import Batch

    torch.manual_seed(0)
    model = two_tower.MultiTower(torch.randn(20, 6), 6.0, embedding_size=8, hidden_size=16, n_interests=3)
    batch = Batch(torch.randint(0, 20, (4, 7)), torch.rand(4, 7) * 10, torch.ones(4, 7, dtype=torch.bool), None, None, None)
    # with strong, different interests, each looks at different games in the history
    model.interests.data *= 300
    assert float(interest_overlap(model, batch)) < 0.99
    model.interests.data[:] = model.interests.data[0]
    assert float(interest_overlap(model, batch)) == pytest.approx(1.0, abs=1e-5)
