# bgg-recommender

A neural network recommender for BoardGameGeek, built in PyTorch. It picks up where the content-based model in [traditional-recommenders](https://github.com/yorker78/traditional-recommenders) left off: the features were designed there, but the model never trained.

**Status: in progress.** The data pipeline, training loop, and evaluation are done. The model itself is next.

## Scores to beat

Every model is scored on the same held-out users: a fifth of the users never appear in training, and for each of them the model sees 70% of their ratings and predicts the other 30%. R2 is the squared correlation between predicted and actual ratings, the same measure the earlier notebooks used. MAE is the average miss in rating points, on BoardGameGeek's 1 to 10 scale.

| Model | R2 | MAE | Test ratings predicted |
| - | - | - | - |
| Average rating of each game | 0.2588 | | 1,144,841 |
| kNN, 500 neighbors (from traditional-recommenders) | 0.2689 | | 1,070,377 |
| Game averages, learned by the training loop | 0.2517 | 1.009 | 1,144,841 |
| Game averages plus a per-user offset | 0.4169 | 0.881 | 1,144,841 |

The last row is the real bar. It only knows whether a user tends to rate above or below each game's average, and that alone explains far more than kNN did, because the kNN notebook averaged its neighbors' raw ratings without accounting for how generous each user is.

The kNN results are the ones saved by the traditional-recommenders notebook, which used this same split. kNN can only predict a rating when one of the user's neighbors rated that game, so it covers fewer test ratings.

## The model interface

A model is a `torch.nn.Module` that never sees a user id. It is given each user as a history of rated games and predicts their ratings for other games:

```python
def forward(self, history_items, history_ratings, history_mask, target_items):
    # history_items   [users, history]  game indices
    # history_ratings [users, history]  the ratings they gave those games
    # history_mask    [users, history]  False where a shorter history is padded
    # target_items    [users, targets]  games to predict for
    # returns         [users, targets]  predicted ratings
```

`bgg/baselines.py` has the smallest example. Because a model is never handed the ratings it is asked to predict, it cannot leak them.

## Running it

Download the [BoardGameGeek board games database](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) and put its CSV files in `data/`. Install [PyTorch](https://pytorch.org/get-started/locally/) for your machine separately, since the right build depends on your GPU.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
.venv/bin/python -m bgg.features    # writes the game feature and creator tables
.venv/bin/python -m bgg.evaluate    # prints the average-rating score
.venv/bin/python -m bgg.train --name game-averages --model bias --lr 0.05
```

## Layout

- `bgg/data.py` loads the ratings and splits them into training, validation, query, and test sets.
- `bgg/features.py` builds one row of features per game (complexity, player counts, playtime, mechanics, themes, and a few engineered ones) and the list of each game's designers, artists, and publishers.
- `bgg/batches.py` turns ratings into batches of users, each with a history and targets.
- `bgg/baselines.py` has the reference models.
- `bgg/train.py` trains a model and appends its scores to `results.jsonl`.
- `bgg/evaluate.py` scores predictions and computes the average-rating baseline.
