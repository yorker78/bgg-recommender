# bgg-recommender

This is a Board Game Recommendation System using a two-tower deep learning architecture to form the recommendations. I had previously attempted making a simpler MLP based Recommendation System for my graduate study on this topic, but during the course I sidetracked myself chasing down a memory leak in polars and ran out of time.

It has two models, both in PyTorch: one that predicts the rating a user would give a game, and a two-tower model that recommends games, which is the one behind the demo. **[Try the demo](https://www.billyyork.com/bgg-recommender/)**: pick a few games you like and it recommends others, running entirely in your browser. More context is on my portfolio at [billyyork.com](https://www.billyyork.com).

## Results

Every model is scored on users it never saw in training: a fifth of the users are held out, and for each of them the model is given 70% of their ratings and judged on the other 30%. Scores are the mean of three seeds on all 61,131 held-out users.

### Recommending

A recommendation counts as a hit when it is a game the user went on to rate 8 or higher. Each row adds one change to the row above it.

| Model | Liked games in the top 10 | Recall@50 | Hits beyond the 100 best-known games | Recommendations among those 100 | Different games recommended |
| --- | --- | --- | --- | --- | --- |
| Most popular games | 0.635 | 0.285 | 0.000 | 100% | 49 |
| Two-tower | 0.898 | 0.402 | 0.208 | 75% | 1,266 |
| + game descriptions and ratios | 0.920 | 0.409 | 0.226 | 73% | 1,321 |
| + negatives drawn by popularity | 0.996 | 0.421 | 0.267 | 68% | 2,175 |
| + a learned vector per game family | 1.032 | 0.435 | 0.308 | 65% | 2,832 |
| **+ tuned (1,024 negatives, learning rate 0.003)** | **1.088** | **0.441** | 0.303 | 71% | 2,093 |

The final model finds about 1.7 times as many liked games as recommending the most popular ones, and a third of a hit per user comes from outside the 100 best-known games, where popularity can never score. Tuning bought accuracy at some cost in variety.

### Predicting ratings

| Model | R2 | MAE |
| --- | --- | --- |
| Average rating of each game | 0.2588 | |
| kNN, 500 neighbors (from [traditional-recommenders](https://github.com/yorker78/traditional-recommenders)) | 0.2689 | |
| Game averages plus a per-user offset | 0.4169 | 0.881 |
| Rating model with attention | 0.4857 | 0.822 |
| **+ designers, artists, and publishers** | **0.4980** | **0.810** |

R2 is the squared correlation between predicted and actual ratings, the measure the earlier coursework used, and MAE is the average miss on BoardGameGeek's 1 to 10 scale. A good rating predictor turned out to be a poor recommender: it found 0.078 liked games in its top 10, far below simply recommending popular games, which is why the recommender is a separate model trained to rank.

## How it works

- **A user is the games they have rated.** No model ever sees a user id, so it works for anyone, including a visitor to the demo.
- **Game vectors** come from each game's features: complexity, player counts, playtime, mechanics, themes, a TF-IDF and SVD summary of its description, a few engineered ratios, and a learned vector shared by games in the same family.
- **The user tower** averages the vectors of the games a user rated, weighted by how much they liked each one, and turns that into one "ideal game". Recommendations are the games closest to it by cosine similarity.
- **Training** pushes each user's liked games above games drawn at random. Drawing those in proportion to popularity, so that well-known games also turn up as examples of what a user did not pick, made the biggest single difference.
- **The demo** exports the game vectors and the user tower's weights, and repeats the same arithmetic in JavaScript, checked against PyTorch by the tests in `demo/`.

## Running it

Download the [BoardGameGeek board games database](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) and put its CSV files in `data/`. Install [PyTorch](https://pytorch.org/get-started/locally/) for your machine separately, since the right build depends on your GPU.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
.venv/bin/python -m bgg.features    # writes the game feature, creator, and family tables

# the final recommender, then its scores on every held-out user
.venv/bin/python -m bgg.train --name final --model two-tower --objective ranking --cosine \
    --use-descriptions --use-ratios --use-family --negative-popularity 0.25 \
    --negatives 1024 --lr 0.003 --epochs 100 --patience 8
.venv/bin/python -m bgg.retrieval popularity final --max-users 1000000

# the best rating model
.venv/bin/python -m bgg.train --name rating --model recommender --use-content --use-creators \
    --use-seen --use-context --use-attention --hidden-sizes 64 --lr 0.003 --epochs 200 --patience 8

# the demo: export a trained recommender, then serve the page
.venv/bin/python -m bgg.export final
node --test demo/recommend.test.js demo/select.test.js
cd demo && python -m http.server
```

Training appends each run's settings and scores to `results.jsonl` and saves its weights in `runs/`.

## Layout

- `bgg/data.py` loads the ratings and splits them into training, validation, query, and test sets.
- `bgg/features.py` builds one row of features per game, and each game's creators and family.
- `bgg/batches.py` turns ratings into batches of users, each with a history and targets.
- `bgg/baselines.py` has the reference models.
- `bgg/model.py` is the rating model, and `bgg/two_tower.py` the recommender, with `MultiTower`, a variant with several ideal games per user.
- `bgg/multi_interest.py` clusters a user's history into several ideal games after training, an experiment that did not help.
- `bgg/train.py` trains a model and records its scores, `bgg/evaluate.py` scores rating predictions, and `bgg/retrieval.py` scores recommendations.
- `bgg/export.py` writes a trained recommender as the files the demo loads.
- `demo/` is the demo page, and `.github/workflows/demo.yml` publishes it.

## Known limitations

- **The data ends in 2021.** Newer games are missing, which the first people to try the demo noticed straight away.
- **It still leans on popular games.** About 71% of its recommendations are among the 100 best-known games, though they are no longer all of them.
- **It is weaker with few games to go on.** It was tested on users with many ratings; a visitor who picks three or five games gets a less personal list.
- **It only learns from ratings.** It has no signal for a game someone owns but never rated, or one they would like to try.

## Future work

- The same architecture on other kinds of ratings, such as books or movies. Each dataset needs its own features, but the central table of users, items, and ratings stays the same.
- Signals beyond ratings, such as wishlists, better inference of what a user dislikes, and "played it but never rated it".

## LLM use

I wrote the various models, the export script, and the in-browser scoring system. Claude built the training and evaluation systems around them. We both contributed to feature engineering efforts. I felt this was fine as I have written training loops, scorers, experimentation setups, and other deep learning support systems in a few of my classes, but I wanted to focus on the model itself and the decisions which go into its architecture. Claude and I collaborated and brainstormed about the choices, with it providing a few ideas that worked, a few that didn't, as well as helping with debugging dimensionality mismatches along the way. As the architecture became more complex, I leaned on Claude more for help with designs unfamiliar to me. Claude also wrote the tests and the demo page, and prototyped ideas in a separate sandbox before I built the ones worth building.

## Data and licensing

The ratings and game details come from the [Board Games Database from BoardGameGeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek), published on Kaggle by threnjen under the [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license. Its description says it was sourced and shared in accordance with BoardGameGeek's XML API Terms of Use. This project does not call the BoardGameGeek API.

This repository is the source for the whole project, including the parts that are hosted elsewhere:

- **Code**, here: MIT license.
- **Trained model weights and the demo**, built from the dataset above and shared under CC BY-SA 3.0 with credit to it. That includes the game names and details the demo displays. The demo and its exported model are published from this repository's GitHub Pages site; see [LICENSE-ARTIFACTS](LICENSE-ARTIFACTS).
- **The dataset itself** is not redistributed. Download it from Kaggle to run the code.
