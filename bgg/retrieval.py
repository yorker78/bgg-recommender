"""Scoring a model as a recommender rather than as a rating predictor.

    python -m bgg.retrieval popularity game-averages content-head64-seen
    python -m bgg.retrieval two-tower-cosine --cluster kmeans:2 hdbscan --by-taste

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
from torch import Tensor

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
    only: Tensor | None = None,
    rating_counts: Tensor | None = None,
) -> dict[str, float]:
    """Averages over users who have at least one liked test game:

    - hits@10: how many of the top 10 recommendations are liked test games
    - recall@50: the share of their liked test games found in the top 50
    - ndcg@10: like hits@10, but a hit counts for more the higher it is ranked,
      scaled so that 1.0 is the best possible top 10

    `only` is a true/false value per user; when given, just those users are scored.

    With `rating_counts` (how many ratings each game has), three more numbers
    say how adventurous the recommendations are. "Well known" means among the
    100 most-rated games.

    - beyond_hits@10: liked test games in the top 10 that are not well known,
      the hits a list of popular games could never have made
    - well_known@10: the share of all top 10 recommendations that are well known
    - games_recommended: how many different games appear in anyone's top 10
    """
    model.eval()
    device = test.items.device
    user_of = torch.repeat_interleave(torch.arange(len(test.counts), device=device), test.counts)
    has_liked = torch.zeros(len(test.counts), dtype=torch.bool, device=device)
    has_liked[user_of[test.ratings >= liked]] = True
    if only is not None:
        has_liked &= only
    users = has_liked.nonzero().squeeze(1)
    if len(users) > max_users:
        pick = torch.randperm(len(users), generator=torch.Generator().manual_seed(seed))[:max_users]
        users = users[pick.to(device)].sort().values

    all_games = torch.arange(n_games, device=device)
    wide = min(50, n_games)
    discount = 1 / torch.log2(torch.arange(2, 12, device=device).float())  # ranks 1 to 10
    best_dcg = discount.cumsum(0)
    totals = {"hits@10": 0.0, "recall@50": 0.0, "ndcg@10": 0.0}
    if rating_counts is not None:
        well_known = torch.zeros(n_games, dtype=torch.bool, device=device)
        well_known[rating_counts.to(device).topk(min(100, n_games)).indices] = True
        recommended = torch.zeros(n_games, dtype=torch.bool, device=device)
        totals |= {"beyond_hits@10": 0.0, "well_known@10": 0.0}
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

        top = scores.topk(wide).indices
        hit = relevant.gather(1, top).float()
        n_relevant = relevant.sum(1)
        keep = n_relevant > 0
        hit, n_relevant = hit[keep], n_relevant[keep]
        if rating_counts is not None:
            top10 = top[keep][:, :10]
            # with very few games, a top 10 can reach down into the excluded ones
            real = scores[keep].gather(1, top10).isfinite()
            familiar = well_known[top10] & real
            totals["beyond_hits@10"] += float((hit[:, :10] * ~familiar).sum())
            totals["well_known@10"] += float((familiar.sum(1) / real.sum(1).clamp(min=1)).sum())
            recommended[top10[real]] = True
        totals["hits@10"] += float(hit[:, :10].sum())
        totals["recall@50"] += float((hit.sum(1) / n_relevant).sum())
        dcg = (hit[:, :10] * discount[: hit[:, :10].shape[1]]).sum(1)
        totals["ndcg@10"] += float((dcg / best_dcg[n_relevant.clamp(max=10) - 1]).sum())
        scored += int(keep.sum())
    result = {name: total / scored for name, total in totals.items()} | {"users": scored}
    if rating_counts is not None:
        result["games_recommended"] = int(recommended.sum())
    return result


def taste_variety(query: RatingTable, categories: Tensor, liked: float = 8.0, min_liked: int = 10) -> Tensor:
    """How spread out each user's liked games are across BoardGameGeek's
    categories (strategy, war, party, family and so on), from their query
    ratings: 0 when every liked game is in one category, higher the more evenly
    they are spread. NaN for a user with fewer than `min_liked` liked games,
    since a user with two liked games looks focused only for lack of evidence.

    `categories` is [games, categories] of 0s and 1s. This is the entropy of
    the user's share of liked games per category.
    """
    user_of = torch.repeat_interleave(torch.arange(len(query.counts), device=query.items.device), query.counts)
    is_liked = query.ratings >= liked
    counts = torch.zeros(len(query.counts), categories.shape[1], device=query.items.device)
    counts.index_add_(0, user_of[is_liked], categories[query.items[is_liked]].float())
    share = counts / counts.sum(1, keepdim=True)
    variety = -(share * torch.log(share.clamp(min=1e-12))).sum(1)
    n_liked = torch.zeros(len(query.counts), device=query.items.device).index_add_(0, user_of[is_liked], torch.ones_like(user_of[is_liked], dtype=torch.float))
    return torch.where((n_liked >= min_liked) & (counts.sum(1) > 0), variety, torch.nan)


if __name__ == "__main__":
    # imported here because bgg.train uses score_retrieval
    from bgg.train import PLAYER_COUNT_FEATURES, build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="runs from results.jsonl, or 'popularity'")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--cluster", nargs="*", default=[], metavar="METHOD[:K]",
                        help="also score each model with one ideal vector per taste cluster, e.g. kmeans:2 kmeans-auto agglomerative hdbscan dbscan categories")
    parser.add_argument("--by-taste", action="store_true",
                        help="also report users with the most focused and most varied tastes separately")
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
        counts = index.join(train.group_by("item_id").len(), on="item_id", how="left").sort("item")["len"].fill_null(0).to_torch()
        if name == "popularity":
            model = Popularity(counts)
        else:
            model = build_model(settings, 0.0, features, creators, n_creator_ids)
            state = torch.load(Path("runs") / f"{name}.pt", map_location="cpu")
            # strict=False: older runs predate the creator_mask buffer, which is rebuilt anyway
            problems = model.load_state_dict(state, strict=False)
            assert not problems.unexpected_keys and set(problems.missing_keys) <= {"creator_mask"}, problems
        query_table, test_table = (to_table(part, index, users).to(device) for part in (query, test))
        model.to(device)

        # BoardGameGeek's own categories: strategy, war, party, family and so on
        _, raw, names, _, _ = load_game_tensors(args.data_dir)
        categories = raw[:, [n.startswith("Cat:") for n in names]].long().to(device)

        variants = {name: model}
        for spec in args.cluster:
            from bgg.multi_interest import MultiInterest

            method, _, k = spec.partition(":")
            variants[f"{name} + {spec}"] = MultiInterest(
                model, method=method, n_clusters=int(k or 2), categories=categories
            ).to(device)

        groups = {"": None}
        if args.by_taste:
            variety = taste_variety(query_table, categories)
            low, high = variety[~variety.isnan()].quantile(torch.tensor([1 / 3, 2 / 3], device=device))
            groups |= {
                " (enough history to tell)": ~variety.isnan(),
                " (focused tastes)": variety <= low,
                " (varied tastes)": variety >= high,
            }

        for label, variant in variants.items():
            for group, only in groups.items():
                result = score_retrieval(
                    variant, query_table, test_table, len(item_ids),
                    max_users=args.max_users, only=only, rating_counts=counts,
                )
                print(
                    f"{label}{group}: {result['hits@10']:.3f} liked games in the top 10, "
                    f"recall@50 {result['recall@50']:.4f}, NDCG@10 {result['ndcg@10']:.4f} ({result['users']:,} users)\n"
                    f"    {result['beyond_hits@10']:.3f} of those hits are beyond the 100 best-known games, "
                    f"{result['well_known@10']:.0%} of recommendations are among those 100, "
                    f"{result['games_recommended']:,} different games recommended",
                    flush=True,
                )
