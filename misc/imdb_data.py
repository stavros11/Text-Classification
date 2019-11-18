import os
import re
import pandas as pd
import tensorflow as tf
from typing import Tuple


def load_directory_data(directory: str) -> pd.DataFrame:
  """Loads all files from a directory in a DataFrame."""
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with open(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)


def load_dataset(directory: str) -> pd.DataFrame:
  """Merge positive and negative examples, add a polarity column and shuffle."""
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Download and process IMBDb dataset."""
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz",
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
      extract=True)

  train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      "aclImdb", "test"))

  return train_df, test_df


def load_datasets_locally(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Load and process IMBDb dataset from local directory."""
  train_df = load_dataset(os.path.join(data_dir, "aclImdb", "train"))
  test_df = load_dataset(os.path.join(data_dir, "aclImdb", "test"))
  return train_df, test_df