"""Define keras models for easier import in notebooks."""
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from typing import List, Optional, Tuple


def create_one_hot_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
  """Create one hot labels from any type (possibly str) labels.

  Args:
    labels: Labels of arbitrary type.

  Returns:
    y: Array of the same length as `labels` that contains the one-hot
      representations.
    classes: List with all the different class identities. The order of this
      list agrees with the ordering of 1s in one hot representations.
  """
  classes = np.unique(labels)
  y = np.zeros((len(labels), len(classes)))
  for i, c in enumerate(classes):
    ind = np.where(labels == c)[0]
    y[ind, i] = 1
  return y, classes


def simple_embedding_model(num_words: int,
                           embedding_dims: int,
                           num_classes: int,
                           use_one_hot: bool = True,
                           max_len: Optional[int] = None) -> models.Model:
  """Simple word embedding model with averaging and a dense output.

  Args:
    num_words: Number of words in the vocabulary needed for creating the
      embedding layer.
    embedding_dims: Embedding vector dimension.
    num_classes: Number of classification classes for the last layer dimension.
    use_one_hot: Relevant only for `num_classes` > 2.
    max_len: Maximum length of sentence
      (generally not required since we do average pooling before the dense).

  Returns:
    Keras model that does these transformations.
  """
  model = models.Sequential()
  model.add(layers.Embedding(num_words, embedding_dims, input_length=max_len))
  model.add(layers.GlobalAveragePooling1D())
  if num_classes > 2:
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
  # If we have binary classification we can choose between one-hot or
  # simply a sigmoid.
  if use_one_hot:
    model.add(layers.Dense(num_classes, activation='softmax'))
  else:
    model.add(layers.Dense(1, activation='sigmoid'))
  return model