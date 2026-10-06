"""Scoring a model as a recommender rather than as a rating predictor.

    python -m bgg.retrieval popularity game-averages content-head64-seen

For each held-out user the model is given their query ratings and ranks every
game that is not among them. It is scored on where the games the user went on
to like (test ratings of `liked` or more) land in that ranking.

A game the user never rated counts as a miss even if they would love it, so
every model looks worse here than it is. The numbers are for comparing models.
"""

import argparse
import json
from pathlib import Path

import torch

from bgg.baselines import Popularity
from bgg.batches import RatingTable, _gather, game_index, to_table
from bgg.data import hold_out_validation_users, load_ratings, split
from bgg.features import load_game_tensors


@torch.no_grad()
def score_retrieval(
    model: torch.nn.Module,
    query: RatingTable,
    test: RatingTable,
    n_games: int,
    liked: float = 8.0,
    users_per_batch: int = 32,
    max_history: int = 256,
    # ponytail: a fixed sample of users keeps this to a minute or two; raise it
    # if differences between models get smaller than the sampling noise
    max_users: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Averages over users who have at least one liked test game:

    - hits@10: how many of the top 10 recommendations are liked test games
    - recall@50: the share of their liked test games found in the top 50
    - ndcg@10: like hits@10, but a hit counts for more the higher it is ranked,
      scaled so that 1.0 is the best possible top 10
    """
    model.eval()
    device = test.items.device
    user_of = torch.repeat_interleave(torch.arange(len(test.counts), device=device), test.counts)
    has_liked = torch.zeros(len(test.counts), dtype=torch.bool, device=device)
    has_liked[user_of[test.ratings >= liked]] = True
    users = has_liked.nonzero().squeeze(1)
    if len(users) > max_users:
        pick = torch.randperm(len(users), generator=torch.Generator().manual_seed(seed))[:max_users]
        users = users[pick.to(device)].sort().values

    all_games = torch.arange(n_games, device=device)
    wide = min(50, n_games)
    discount = 1 / torch.log2(torch.arange(2, 12, device=device).float())  # ranks 1 to 10
    best_dcg = discount.cumsum(0)
    totals = {"hits@10": 0.0, "recall@50": 0.0, "ndcg@10": 0.0}
    scored = 0
    for batch in users.split(users_per_batch):
        rows = torch.arange(len(batch), device=device)[:, None]
        n_history = query.counts[batch].clamp(max=max_history)
        history = _gather(query, query.offsets[batch], n_history, max(int(n_history.max()), 1))
        scores = model(*history, all_games.expand(len(batch), -1))

        # never recommend a game from their history, including any past max_history
        rated, _, is_rated = _gather(query, query.offsets[batch], query.counts[batch], max(int(query.counts[batch].max()), 1))
        excluded = torch.zeros_like(scores, dtype=torch.bool)
        excluded[rows.expand_as(rated)[is_rated], rated[is_rated]] = True
        scores = scores.masked_fill(excluded, float("-inf"))

        targets, ratings, is_target = _gather(test, test.offsets[batch], test.counts[batch], int(test.counts[batch].max()))
        is_liked = is_target & (ratings >= liked)
        relevant = torch.zeros_like(excluded)
        relevant[rows.expand_as(targets)[is_liked], targets[is_liked]] = True
        relevant &= ~excluded  # a few users rated the same game twice

        hit = relevant.gather(1, scores.topk(wide).indices).float()
        n_relevant = relevant.sum(1)
        keep = n_relevant > 0
        hit, n_relevant = hit[keep], n_relevant[keep]
        totals["hits@10"] += float(hit[:, :10].sum())
        totals["recall@50"] += float((hit.sum(1) / n_relevant).sum())
        dcg = (hit[:, :10] * discount[: hit[:, :10].shape[1]]).sum(1)
        totals["ndcg@10"] += float((dcg / best_dcg[n_relevant.clamp(max=10) - 1]).sum())
        scored += int(keep.sum())
    return {name: total / scored for name, total in totals.items()} | {"users": scored}


if __name__ == "__main__":
    # imported here because bgg.train uses score_retrieval
    from bgg.train import PLAYER_COUNT_FEATURES, build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="runs from results.jsonl, or 'popularity'")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-users", type=int, default=10_000)
    args = parser.parse_args()
    device = torch.device(args.device)

    runs = {}
    for line in open("results.jsonl"):
        record = json.loads(line)
        runs[record["name"]] = record["settings"]  # the latest run of a name wins, as its weights did
    # settings a run may predate
    defaults = dict(use_seen=False, use_context=False, use_attention=False, embedding_size=32, hidden_sizes=[], cosine=False)

    train, query, test = split(load_ratings(args.data_dir))
    train, _, _ = hold_out_validation_users(train)
    users = test["user_id"].unique().sort()
    for name in args.names:
        settings = argparse.Namespace(**(defaults | runs.get(name, {})))
        drop = PLAYER_COUNT_FEATURES if getattr(settings, "drop_player_counts", False) else ()
        item_ids, features, _, creators, n_creator_ids = load_game_tensors(args.data_dir, drop)
        index = game_index(item_ids)
        if name == "popularity":
            counts = index.join(train.group_by("item_id").len(), on="item_id", how="left").sort("item")["len"]
            model = Popularity(counts.fill_null(0).to_torch())
        else:
            model = build_model(settings, 0.0, features, creators, n_creator_ids)
            state = torch.load(Path("runs") / f"{name}.pt", map_location="cpu")
            # strict=False: older runs predate the creator_mask buffer, which is rebuilt anyway
            problems = model.load_state_dict(state, strict=False)
            assert not problems.unexpected_keys and set(problems.missing_keys) <= {"creator_mask"}, problems
        tables = (to_table(part, index, users).to(device) for part in (query, test))
        result = score_retrieval(model.to(device), *tables, len(item_ids), max_users=args.max_users)
        print(
            f"{name}: {result['hits@10']:.3f} liked games in the top 10, "
            f"recall@50 {result['recall@50']:.4f}, NDCG@10 {result['ndcg@10']:.4f} ({result['users']:,} users)",
            flush=True,
        )
