"""Applies text preprocessing and saves the generated `DataFrame`."""
import argparse
import os
import numpy as np
import pandas as pd
from utils import preprocessing

# TODO: Possible parallelization of this

parser = argparse.ArgumentParser()
parser.add_argument("--area", default="nyc", type=str,
                    help="Area to use data from (must have the data downloaded).")
parser.add_argument("--data-dir", default="/home/stavros/DATA/AirbnbReviews",
                    type=str, help="Directory path to the data files.")
parser.add_argument("--samples", default=None, type=int,
                    help="Number of samples to use (to make it faster than using all data.")
parser.add_argument("--message", default=None, type=int,
                    help="Every how many processed texts to print messages.")


def main(data_dir, area, samples=None, message=None):
  area_dir = os.path.join(data_dir, area)
  data = pd.read_csv(os.path.join(area_dir, "reviews.csv.gz"))

  # Remove lines for which reviews are nan
  clean_data = data[pd.notnull(data.comments)]

  if samples is not None:
    ids = np.random.randint(0, len(clean_data), samples)
    # Remember that using `.loc` instead of `.iloc` here FAILS!
    sampled_data = clean_data.iloc[ids]
  else:
    sampled_data = clean_data

  # Check if sampled data are all strings (and not `nan` etc.)
  for doc in sampled_data.comments:
    if not isinstance(doc, str):
      raise TypeError("{}".format(doc))

  print("Loaded {} reviews.".format(area))
  print(sampled_data.shape)
  normalizer = preprocessing.CorpusNormalizer(special_char_removal=True,
                                              remove_digits=True,
                                              text_lemmatization=True,
                                              stopword_removal=False)
  sampled_data["normalized_comments"] = normalizer(sampled_data.comments,
              n_message=message)
  sampled_data.to_csv("{}_reviews_{}samples.csv".format(area, samples))


if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))