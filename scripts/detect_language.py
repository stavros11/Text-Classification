"""Detects language of reviews using `langdetect`.

Creates a new column "language" in the DataFrame.
"""
import argparse
import os
import pandas as pd
import langdetect
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument("--area", default="nyc", type=str,
                    help="Area to use data from (must have the data downloaded).")
parser.add_argument("--data-dir", default="/home/stavros/DATA/AirbnbReviews",
                    type=str, help="Directory path to the data files.")
parser.add_argument("--start-ind", default=0, type=int,
                    help="Row index of the first data point.")
parser.add_argument("--finish-ind", default=None, type=int,
                    help="Row index of the first data point.")
parser.add_argument("--message", default=None, type=int,
                    help="Every how many processed texts to print messages.")


def main(data_dir: str, area: str,
         start_ind: int = 0,
         finish_ind: Optional[int] = None,
         message: Optional[int] = None):
  # Load reviews for the given area
  area_dir = os.path.join(data_dir, area)
  data = pd.read_csv(os.path.join(area_dir, "reviews.csv.gz"))
  print("Loaded {} reviews.".format(area))
  # Remove lines for which reviews are nan
  clean_data = data[pd.notnull(data.comments)]

  if finish_ind is None:
    finish_ind = clean_data.shape[0]
  assert finish_ind > start_ind
  print("Detecting language from {} to {}.".format(start_ind, finish_ind))

  new_cols = list(clean_data.columns) + ["comments_language"]
  samples = finish_ind - start_ind
  identified_data = pd.DataFrame(index=range(samples), columns=new_cols)
  ic = 0
  for i in range(start_ind, finish_ind):
    identified_data.iloc[ic] = clean_data.iloc[i]
    review = clean_data.iloc[i]["comments"]
    try:
      identified_data.iloc[ic]["comments_language"] = langdetect.detect(review)
    except:
      print("Failed to detect language for:", review)
      identified_data.iloc[ic]["comments_language"] = "<unk>"
    ic += 1
    if i % message == 0:
      print("{} / {}".format(i, finish_ind))

  # Save new DataFrame
  savename = "{}_reviews_withlang_{}to{}.csv".format(area, start_ind, finish_ind)
  identified_data.to_csv(savename, index=False)


if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))