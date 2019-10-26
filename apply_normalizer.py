"""Applies text preprocessing and saves the generated `DataFrame`."""
import argparse
import os
import numpy as np
import pandas as pd
import langdetect
from utils import preprocessing
from typing import Optional

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

# Note that the following three options are enabled by default for convenience
#parser.add_argument("--remove-stopwords", action="store_true",
#                    help="Whether to remove stopwords from the corpus.")
#parser.add_argument("--lemmatize", action="store_true",
#                    help="Whether to lemmatize words in the corpus.")
#parser.add_argument("--english-only", action="store_true",
#                    help="Whether to keep only english reviews.")


def main(data_dir: str, area: str,
         samples: Optional[int] = None,
         remove_stopwords: bool = True,
         lemmatize: bool = True,
         english_only: bool = True,
         message: Optional[int] = None):
  # Load reviews for the given area
  area_dir = os.path.join(data_dir, area)
  data = pd.read_csv(os.path.join(area_dir, "reviews.csv.gz"))
  print("Loaded {} reviews.".format(area))
  # Remove lines for which reviews are nan
  clean_data = data[pd.notnull(data.comments)]

  if samples is None:
    samples = clean_data.shape[0]

  normalizer = preprocessing.CorpusNormalizer(special_char_removal=True,
                                              remove_digits=True,
                                              text_lemmatization=lemmatize,
                                              stopword_removal=remove_stopwords)
  print("\nText normalizer has the following functionalities:")
  print(normalizer)

  if english_only:
    print("\nKeeping reviews in english only.")

  # Generate indices to randomly loop over data
  ids = np.arange(clean_data.shape[0])
  np.random.shuffle(ids)

  # Create new DataFrame to fill with sampled data points
  sampled_columns = list(clean_data.columns) + ["normalized_comments"]
  sampled_data = pd.DataFrame(index=range(samples), columns=sampled_columns)
  # Fill with valid datapoints
  i, ic = 0, 0
  while ic < samples:
    data_row = clean_data.iloc[i]
    review = data_row["comments"]
    i += 1
    # A review is valid if it is a string and has more than 5 characters
    is_valid = isinstance(review, str) and len(review) > 5
    if is_valid and english_only:
      try:
        is_english = langdetect.detect(data_row["comments"]) == "en"
      except:
        print(data_row["comments"])
        continue
    else:
      is_english = True
    if is_valid and is_english:
      sampled_data.iloc[ic] = data_row
      ic += 1

    if english_only and ic % message == 0:
      print("{} / {} english reviews found.".format(ic, samples))

  print("\nSampled data created with shape", sampled_data.shape)
  sampled_data["normalized_comments"] = normalizer(sampled_data.comments,
              n_message=message)

  # Save new DataFrame
  stopwords = ["", "_nostopwords"][int(remove_stopwords)]
  english = ["", "_en"][int(english_only)]
  savename = "{}_reviews{}{}_{}samples.csv".format(
      area, stopwords, english, samples)
  sampled_data.to_csv(savename)


if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))