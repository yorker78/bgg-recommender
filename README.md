# bgg-recommender

A neural network recommender for BoardGameGeek, built in PyTorch. It picks up where the content-based model in [traditional-recommenders](https://github.com/yorker78/traditional-recommenders) left off: the features were designed there, but the model never trained.

**Status: in progress.** The data pipeline and evaluation are done. The model is next.

## Scores to beat

Every model is scored on the same held-out users: a fifth of the users never appear in training, and for each of them the model sees 70% of their ratings and predicts the other 30%. The score is the squared correlation (R2) between predicted and actual ratings, the same measure the earlier notebooks used.

| Model | R2 | Test ratings predicted |
| - | - | - |
| Average rating of each game | 0.2588 | 1,144,841 |
| kNN, 500 neighbors (from traditional-recommenders) | 0.2689 | 1,070,377 |

The kNN results are the ones saved by the traditional-recommenders notebook, which used this same split. kNN can only predict a rating when one of the user's neighbors rated that game, so it covers fewer test ratings.

## Running it

Download the [BoardGameGeek board games database](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) and put `user_ratings.csv`, `games.csv`, `mechanics.csv`, `themes.csv`, and `subcategories.csv` in `data/`.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
.venv/bin/python -m bgg.features    # writes data/game_features.parquet
.venv/bin/python -m bgg.evaluate    # prints the average-rating score
```

## Layout

- `bgg/data.py` loads the ratings and splits them into training, query, and test sets.
- `bgg/features.py` builds one row of features per game: complexity, player counts, playtime, mechanics, themes, and a few engineered ones, like how many of a game's supported player counts the community actually recommends.
- `bgg/evaluate.py` scores predictions and computes the average-rating baseline.
