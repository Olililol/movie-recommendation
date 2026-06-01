import sys
import numpy as np
import pandas as pd
import urllib.request
import os
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import CTR
from cornac.metrics import NDCG, Recall
from cornac.data import TextModality

# fetch data and format it
data = cornac.datasets.movielens.load_feedback(variant='100K')

# fetch movie plots (content-based part)
plots, movie_ids = cornac.datasets.movielens.load_plot()
item_text = TextModality(corpus=plots, ids=movie_ids)

# fetch movie titles (guard the download so a network failure is reported clearly)
if not os.path.exists("u.item"):
    try:
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-100k/u.item", "u.item"
        )
    except Exception as e:
        sys.exit(f"Failed to download u.item (needed for movie titles): {e}")

movies = pd.read_csv("u.item", sep="|", encoding="latin-1",
                     usecols=[0, 1], names=["movieId", "title"])
# build dictionary
id_to_title = dict(zip(movies["movieId"].astype(str), movies['title']))


# Split data into train/test sets
eval_method = RatioSplit(data, test_size=0.2, rating_threshold=4.0,
                         item_text=item_text, seed=123)
# CTR - hybrid recommendation
ctr = CTR(k=10, max_iter=100, seed=123)


cornac.Experiment(
    eval_method=eval_method,
    models=[ctr],
    metrics=[NDCG(k=10), Recall(k=10)]
).run()


def sample_recommendation(model, eval_method, id_to_title, user_ids, top_n=3):
    """Print known-liked and recommended titles for the given raw user IDs.

    user_ids are raw MovieLens user IDs (as in the original data), not Cornac's
    internal indices. They are mapped to internal indices via uid_map.
    """
    train_set = eval_method.train_set
    item_ids = np.array(train_set.item_ids)        # raw item IDs in internal order
    user_id_map = train_set.uid_map                # raw user ID (str) -> internal index
    rating_matrix = train_set.matrix.tocsr()

    for user_id in user_ids:
        raw_uid = str(user_id)
        if raw_uid not in user_id_map:
            print("User %s not found in the training set; skipping." % raw_uid)
            continue

        user_idx = user_id_map[raw_uid]

        # this user's rated items in training, with the ratings they gave
        user_row = rating_matrix[user_idx]
        seen_idx = user_row.indices
        seen_ratings = user_row.data

        # user's top-N known positives, ranked by the rating they gave
        order = np.argsort(-seen_ratings, kind="stable")
        top_known_idx = seen_idx[order][:top_n]
        top_known_ratings = seen_ratings[order][:top_n]

        # score every item, mask items already seen, then rank
        scores = np.asarray(model.score(user_idx), dtype=float).copy()
        scores[seen_idx] = -np.inf
        ranked = np.argsort(-scores)[:top_n]
        rec_items = item_ids[ranked]
        rec_scores = scores[ranked]

        print("User %s" % raw_uid)
        print("     Top rated (known positives):")
        for idx, rating in zip(top_known_idx, top_known_ratings):
            raw_item = item_ids[idx]
            title = id_to_title.get(str(raw_item).strip(), raw_item)
            print("             %s (rating: %.1f)" % (title, rating))

        print("         Recommended:")
        for raw_item, score in zip(rec_items, rec_scores):
            title = id_to_title.get(str(raw_item).strip(), raw_item)
            print("             %s (predicted score: %.3f)" % (title, score))


# pass id_to_title into the function (raw MovieLens user IDs)
sample_recommendation(ctr, eval_method, id_to_title, [3, 25, 450])