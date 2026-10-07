import argparse
import json
from pathlib import Path

import polars as pl
import torch

from bgg.retrieval import load_run


@torch.no_grad()
def export_demo(run_name, data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, item_ids = load_run(run_name, data_dir)
    model.eval()
    vectors = model.game_vectors(torch.arange(len(item_ids))).cpu().numpy().astype('float32')
    vectors.tofile(out_dir / 'game_vectors.bin')
    w1 = model.user_tower[0].weight.tolist()
    b1 = model.user_tower[0].bias.tolist()
    w2 = model.user_tower[2].weight.tolist()
    b2 = model.user_tower[2].bias.tolist()

    games = pl.DataFrame({'BGGId': item_ids}).join(pl.read_csv(data_dir / 'games.csv', columns=['BGGId', 'Name', 'YearPublished', 'NumUserRatings', 'AvgRating']), on='BGGId', how='left', maintain_order='left')

    with open(out_dir / 'model.json', 'w') as f:
        json.dump({'embedding_size': vectors.shape[1],
                'mean_rating': float(model.mean_rating),
                'tower': {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2},
                'games': {
                    'id': games['BGGId'].to_list(),
                    'name': games['Name'].to_list(),
                    'year': games['YearPublished'].to_list(),
                    'ratings': games['NumUserRatings'].to_list(),
                    'average': games['AvgRating'].to_list()
                }}, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('run_name')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out', default='demo/data')
    args = parser.parse_args()

    export_demo(args.run_name, args.data_dir, args.out)