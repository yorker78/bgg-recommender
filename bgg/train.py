"""Trains a model and records its scores.

    python -m bgg.train --name game-averages --model bias
    python -m bgg.train --name ids-and-content --model recommender --use-ids --use-content
    python -m bgg.train --name two-tower --model two-tower --objective ranking

Every run appends one line to results.jsonl with its settings, its validation
and test scores, and the commit it ran on. The best weights go to runs/.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bgg.baselines import BiasModel
from bgg.batches import game_index, to_table, training_batches
from bgg.data import hold_out_validation_users, load_ratings, split
from bgg.evaluate import score_model
from bgg.features import load_game_tensors
from bgg.retrieval import score_retrieval

PLAYER_COUNT_FEATURES = ("GoodAt", "PlayerCountRatio", "MinPlayers", "MaxPlayers", "BestPlayers")


def build_model(args, train_mean: float, features, creators, n_creator_ids) -> torch.nn.Module:
    if args.model == "bias":
        return BiasModel(features.shape[0], train_mean, user_offset=args.user_offset)
    if args.model == "two-tower":
        from bgg.two_tower import TwoTower

        return TwoTower(
            features,
            train_mean,
            embedding_size=args.embedding_size,
            hidden_size=(args.hidden_sizes or [64])[0],
            cosine=args.cosine,
        )
    if args.model == "multi-tower":
        from bgg.two_tower import MultiTower

        return MultiTower(
            features,
            train_mean,
            embedding_size=args.embedding_size,
            hidden_size=(args.hidden_sizes or [64])[0],
            n_interests=args.interests,
        )
    from bgg.model import Recommender

    return Recommender(
        features,
        creators,
        n_creator_ids,
        train_mean,
        use_ids=args.use_ids,
        use_content=args.use_content,
        use_creators=args.use_creators,
        use_seen=args.use_seen,
        use_context=args.use_context,
        use_attention=args.use_attention,
        embedding_size=args.embedding_size,
        hidden_sizes=tuple(args.hidden_sizes),
    )


def ranking_loss(model, batch, n_games: int, n_negatives: int, generator, liked: float = 8.0):
    """Teaches a model to rank: each user's targets are scored alongside games
    drawn at random, and the model is pushed to score the targets they liked
    (rated `liked` or more) high and everything else low.

    The random games are almost always ones the user never rated. A few will
    be games they did, which is rare enough among 22,000 not to matter.
    """
    n_users = len(batch.target_items)
    negatives = torch.randint(n_games, (n_users, n_negatives), generator=generator, device=batch.target_items.device)
    scores = model(
        batch.history_items, batch.history_ratings, batch.history_mask, torch.cat([batch.target_items, negatives], dim=1)
    )
    wanted = torch.cat([(batch.target_ratings >= liked).float(), torch.zeros_like(negatives, dtype=torch.float)], dim=1)
    real = torch.cat([batch.target_mask, torch.ones_like(negatives, dtype=torch.bool)], dim=1)
    return F.binary_cross_entropy_with_logits(scores[real], wanted[real])


def interest_overlap(model, batch):
    """How alike each user's ideal games are, for a model that makes several:
    the average cosine similarity between two of them, counting a negative one
    as 0. Added to the loss, it keeps the interests from all becoming the same.
    """
    ideals = F.normalize(model.user_vectors(batch.history_items, batch.history_ratings, batch.history_mask), dim=-1)
    alike = ideals @ ideals.transpose(1, 2)  # [users, interests, interests]
    others = ~torch.eye(alike.shape[1], dtype=torch.bool, device=alike.device)
    return alike[:, others].clamp(min=0).mean()


def commit() -> str:
    def git(*command):
        return subprocess.run(["git", *command], capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("+changes" if git("status", "--porcelain") else "")


def main(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device).manual_seed(args.seed)

    drop = PLAYER_COUNT_FEATURES if args.drop_player_counts else ()
    item_ids, features, _, creators, n_creator_ids = load_game_tensors(
        args.data_dir, drop, descriptions=args.use_descriptions, ratios=args.use_ratios
    )
    index = game_index(item_ids)

    train, query, test = split(load_ratings(args.data_dir))
    train, validation_query, validation_test = hold_out_validation_users(train)

    def tables(history, targets):
        users = targets["user_id"].unique().sort()
        return (to_table(part, index, users).to(device) for part in (history, targets))

    train_table = to_table(train, index, train["user_id"].unique().sort()).to(device)
    validation = list(tables(validation_query, validation_test))
    held_out = list(tables(query, test))

    model = build_model(
        args, train["rating"].mean(), features.to(device), creators.to(device), n_creator_ids
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = Path("runs") / f"{args.name}.pt"
    weights.parent.mkdir(exist_ok=True)

    ranking = args.objective == "ranking"
    key = "ndcg@10" if ranking else "r2"

    def evaluate(tables, **options):
        if ranking:
            return score_retrieval(model, *tables, features.shape[0], **options)
        return score_model(model, *tables)

    def describe(scores):
        if ranking:
            return f"{scores['hits@10']:.3f} liked games in the top 10, NDCG@10 {scores['ndcg@10']:.4f}"
        return f"R2 {scores['r2']:.4f}, MAE {scores['mae']:.3f}"

    started = time.time()
    best, epochs_since_best, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in training_batches(
            train_table, args.users_per_batch, args.max_history, args.max_targets, generator
        ):
            if ranking:
                loss = ranking_loss(model, batch, features.shape[0], args.negatives, generator)
                if args.diversity:
                    loss = loss + args.diversity * interest_overlap(model, batch)
            else:
                predicted = model(batch.history_items, batch.history_ratings, batch.history_mask, batch.target_items)
                loss = ((predicted - batch.target_ratings)[batch.target_mask] ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # a smaller sample of users per epoch keeps ranking every game affordable
        scores = evaluate(validation, max_users=2_000) if ranking else evaluate(validation)
        history.append(scores[key])
        print(f"epoch {epoch}: validation {describe(scores)}", flush=True)
        if best is None or scores[key] > best[key]:
            best, epochs_since_best = scores, 0
            torch.save(model.state_dict(), weights)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                break

    model.load_state_dict(torch.load(weights))
    result = {
        "name": args.name,
        "commit": commit(),
        "settings": {k: v for k, v in vars(args).items() if k not in ("name", "data_dir")},
        "epochs": len(history),
        "seconds": round(time.time() - started),
        "validation": best,
        "test": evaluate(held_out),
    }
    with open("results.jsonl", "a") as file:
        file.write(json.dumps(result) + "\n")
    print(f"test {describe(result['test'])}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", choices=["bias", "recommender", "two-tower", "multi-tower"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    # which parts of the model are on
    parser.add_argument("--user-offset", action="store_true")
    parser.add_argument("--use-ids", action="store_true")
    parser.add_argument("--use-content", action="store_true")
    parser.add_argument("--use-creators", action="store_true")
    parser.add_argument("--use-seen", action="store_true")
    parser.add_argument("--use-context", action="store_true")
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--drop-player-counts", action="store_true")
    parser.add_argument("--use-descriptions", action="store_true")
    parser.add_argument("--use-ratios", action="store_true")
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--interests", type=int, default=4)
    # training
    parser.add_argument("--objective", choices=["rating", "ranking"], default="rating")
    parser.add_argument("--negatives", type=int, default=256)
    parser.add_argument("--diversity", type=float, default=0.0,
                        help="for multi-tower: how hard to push a user's ideal games apart")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--users-per-batch", type=int, default=2048)
    parser.add_argument("--max-history", type=int, default=256)
    parser.add_argument("--max-targets", type=int, default=128)
    main(parser.parse_args())
