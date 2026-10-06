"""Trains a model and records its scores.

    python -m bgg.train --name game-averages --model bias
    python -m bgg.train --name ids-and-content --model recommender --use-ids --use-content

Every run appends one line to results.jsonl with its settings, its validation
and test scores, and the commit it ran on. The best weights go to runs/.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from bgg.baselines import BiasModel
from bgg.batches import game_index, to_table, training_batches
from bgg.data import hold_out_validation_users, load_ratings, split
from bgg.evaluate import score_model
from bgg.features import load_game_tensors

PLAYER_COUNT_FEATURES = ("GoodAt", "PlayerCountRatio", "MinPlayers", "MaxPlayers", "BestPlayers")


def build_model(args, train_mean: float, features, creators, n_creator_ids) -> torch.nn.Module:
    if args.model == "bias":
        return BiasModel(features.shape[0], train_mean, user_offset=args.user_offset)
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
        embedding_size=args.embedding_size,
        hidden_sizes=tuple(args.hidden_sizes),
    )


def commit() -> str:
    def git(*command):
        return subprocess.run(["git", *command], capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("+changes" if git("status", "--porcelain") else "")


def main(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device).manual_seed(args.seed)

    drop = PLAYER_COUNT_FEATURES if args.drop_player_counts else ()
    item_ids, features, _, creators, n_creator_ids = load_game_tensors(args.data_dir, drop)
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

    started = time.time()
    best, epochs_since_best, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in training_batches(
            train_table, args.users_per_batch, args.max_history, args.max_targets, generator
        ):
            predicted = model(batch.history_items, batch.history_ratings, batch.history_mask, batch.target_items)
            loss = ((predicted - batch.target_ratings)[batch.target_mask] ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores = score_model(model, *validation)
        history.append(scores["r2"])
        print(f"epoch {epoch}: validation R2 {scores['r2']:.4f}, MAE {scores['mae']:.3f}", flush=True)
        if best is None or scores["r2"] > best["r2"]:
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
        "test": score_model(model, *held_out),
    }
    with open("results.jsonl", "a") as file:
        file.write(json.dumps(result) + "\n")
    print(f"test R2 {result['test']['r2']:.4f}, MAE {result['test']['mae']:.3f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", choices=["bias", "recommender"], required=True)
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
    parser.add_argument("--drop-player-counts", action="store_true")
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[])
    # training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--users-per-batch", type=int, default=2048)
    parser.add_argument("--max-history", type=int, default=256)
    parser.add_argument("--max-targets", type=int, default=128)
    main(parser.parse_args())
