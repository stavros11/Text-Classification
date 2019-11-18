import os
import argparse
from aspects import containers
from utils import directories


def main(filename: str, skip_merging: bool, n_words: int = 200000):
  data_dir = os.path.join(directories.trip_advisor, filename)
  aspects = containers.DataAspects.from_pkl(data_dir)

  if not skip_merging:
    import gensim
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        directories.google_word2vec, binary=True, limit=n_words)
    aspects.create_distance_matrix(word2vec)

  aspects.save()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--filename", type=str)
  parser.add_argument("--skip-merging", action="store_true")
  parser.add_argument("--n-words", type=int, default=200000)

  args = parser.parse_args()
  main(**vars(args))