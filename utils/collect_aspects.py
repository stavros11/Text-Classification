import collections
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import aspect_tools


class AspectCollector:
  """Collects all identified aspects for a particular hotel or listing."""

  def __init__(self, aspects: pd.Series, use_score: bool = False):
    # Some identified aspects are phrases and not single words.
    # Here we transform them to single words
    self.series = aspects
    self.word_series = aspects.map(aspect_tools.make_aspects_single_words)
    # Collect all aspects from the Series to a single Counter
    self.all = aspect_tools.collect_listing_aspects(self.word_series, use_score)

    self.absolute = collections.Counter({k: np.abs(v) for k, v in self.all.items()})

    self.matrix = None
    self.merged = None

  def merge(self, cut_off: float = 0.3, word2vec=None):
    self._cut_off = cut_off
    if self.matrix is None:
      if word2vec is None:
        raise ValueError("Distance matrix not available and word2vec model "
                         "was not given.")
      self.matrix = aspect_tools.DistanceMatrix.calculate(word2vec, self.absolute)

    word_map = self.matrix.word_replacement_map(cut_off=cut_off)
    self.merged = collections.Counter()
    for w, v in self.all.items():
      if w in word_map:
        self.merged[word_map[w]] += v

  @property
  def has_negative(self) -> pd.Series:
    return self.series.map(self._contains_negative)

  @property
  def positive(self) -> collections.Counter:
    return self._pos_or_neg(sign=1)

  @property
  def negative(self) -> collections.Counter:
    return self._pos_or_neg(sign=-1)

  def get_plot_bar(self, start: int = 0, end: int = 20,
                   sign: int = 1):
    aspects = self._pos_or_neg(sign)
    bar_plot_words = aspects.most_common()[start: end]
    bar_plot_words = [word for word, _ in bar_plot_words]
    bar_plot_counts = np.array([aspects[word] for word in bar_plot_words])

    name = "Positive" if sign > 0 else "Negative"
    return go.Bar(y=bar_plot_words, x=bar_plot_counts,
                  orientation="h", name=name)

  @property
  def cut_off(self) -> float:
    return self._cut_off

  def _pos_or_neg(self, sign: int = 1) -> collections.Counter:
    if self.merged is None:
      return collections.Counter(
        {k: sign * v for k, v in self.all.items() if sign * v > 0})

    return collections.Counter(
        {k: sign * v for k, v in self.merged.items() if sign * v > 0})

  @staticmethod
  def _contains_negative(aspects):
    if not aspects:
        return "N/A"
    for v in aspects.values():
        if v < 0: return "Negative"
    return "Positive"