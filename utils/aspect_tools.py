import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple


def collect_aspects(data: pd.DataFrame) -> collections.Counter:
  """Collects all the aspects from a DataFrame.

  The DataFrame must have an `aspects` column that contains sets.
  """
  all_aspects = collections.Counter()
  for aspects in data.aspects:
    for phrase in aspects:
      for word in phrase.split(" "):
        all_aspects[word] += 1
  return all_aspects


def word_barchart(aspects: collections.Counter, n_words: int = 10,
                  n_reviews: Optional[int] = None,
                  plotter=plt, color=None):
  """Plots a bar chart with word frequencies."""
  words_bar = [word for word, _ in aspects.most_common(n_words)]
  if n_reviews is None:
    bar_counts = [count for _, count in aspects.most_common(n_words)]
  else:
    bar_counts = [count / n_reviews for _, count in aspects.most_common(n_words)]
  y_pos = [i for i, _ in enumerate(words_bar)]

  if color is None:
    cp = sns.color_palette()
    color = cp[0]

  plotter.figure(figsize=(10, 6))
  plotter.barh(y_pos, bar_counts, color=color)
  plotter.yticks(y_pos, words_bar)
  plotter.xlabel("Word occurences")
  plotter.show()


class DistanceMatrix:

  def __init__(self, words: List[str],
               distance_matrix: Optional[np.ndarray] = None,
               aspects: Optional[collections.Counter] = None):
    self.words = words
    self.aspects = aspects
    self.matrix = distance_matrix
    if distance_matrix is not None:
      assert len(words) == len(distance_matrix)
      assert distance_matrix.shape[0] == distance_matrix.shape[1]
      assert len(distance_matrix.shape) == 2

  @classmethod
  def calculate(cls, model, aspects: collections.Counter,
                min_appearances: int = 2):
    words = [word for word, counts in aspects.most_common()
             if counts > 2 and word in model]
    print("Calculating matrix with {} words.".format(len(words)))

    matrix = np.eye(len(words))
    for i, word in enumerate(words):
      matrix[i, i:] = model.distances(word, words[i:])

    return cls(words, matrix, aspects)

  def __len__(self) -> int:
    return len(self.words)

  def __getitem__(self, i: int) -> np.ndarray:
    return self.matrix[i]

  @property
  def values(self):
    ids = np.triu_indices(len(self.matrix), k=1)
    return self.matrix[ids]

  def describe(self):
    print("Mean:", self.values.mean())
    print("STD:", self.values.std())
    print("Min:", self.values.min())
    print("Max:", self.values.max())

  def words_closer_than(self, cut_off: float) -> int:
    return (self.values < cut_off).sum()

  def similar_words(self, cut_off: float) -> List[Tuple[str, str]]:
    ids = np.triu_indices(len(self.matrix), k=1)
    ind = np.where(self.matrix[ids] < cut_off)[0]
    sim_words = [(self.words[ids[0][i]], self.words[ids[1][i]]) for i in ind]
    return sim_words

  def merge_words(self, cut_off: float) -> collections.Counter:
    if self.aspects is None:
      raise ValueError("Aspects Counter was not provided.")

    removed_words = set()
    merged_aspects = collections.Counter()

    for i, word in enumerate(self.words):
      if word not in removed_words:
        removed_words.add(word)
        merged_aspects[word] = self.aspects[word]
        for j in np.where(self.matrix[i] < cut_off)[0]:
          if self.words[j] not in removed_words:
            removed_words.add(self.words[j])
            merged_aspects[word] += self.aspects[self.words[j]]

    return merged_aspects


def manual_word_merge(aspects: collections.Counter, word1, word2) -> collections.Counter:
  if word1 not in aspects or word2 not in aspects:
    return aspects

  new_aspects = collections.Counter(aspects)
  new_aspects[word1] += new_aspects.pop(word2)
  return new_aspects