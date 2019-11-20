import os
import collections
import numpy as np
import gensim
from utils import directories
from aspects import containers
from typing import List


hotel_files = ["kresten_royal/the_kresten_royal_villas_1747reviews_withaspects",
               "rodos_palace/rodos_palace_1280reviews_withaspects",
               "sentido_ixian_grand/sentido_ixian_grand_1235reviews_withaspects"]

all_data = [containers.DataAspects.load(data_dir=os.path.join(directories.trip_advisor, hotel_file))
            for hotel_file in hotel_files]

google_vec_file = os.path.join(directories.google_word2vec)
word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True, limit=200000)
print("Word2vec loaded.")


def calculate_distances(phrase: str, categories: List[str]):
  words = phrase.split(" ")
  d = np.stack([word2vec.distances(word, categories) for word in words if word in word2vec])
  return d.mean(axis=0)

def calculate_vectors(phrase: str):
  words = phrase.split(" ")
  d = np.stack([word2vec[word] for word in words if word in word2vec])
  return d.mean(axis=0)


def main(data, categories=["location", "cleanliness", "service", "value"]):
  # Verify that all categories are in model
  for cat in categories:
      assert cat in word2vec

  # Order phrases
  ordered_phrases = []
  for phrase in data.container.words:
      if any([word in word2vec for word in phrase.split(" ")]):
          ordered_phrases.append(phrase)

  print(len(ordered_phrases), len(data.container.words))
  print(len(ordered_phrases) / len(data.container.words))


  categorical_distances = np.array([calculate_distances(phrase, categories) for phrase in ordered_phrases])
  print(np.unique(categorical_distances.argmin(axis=-1), return_counts=True))


  phrase2cat = {phrase: (categories[dist.argmin()], dist.min())
                for phrase, dist in zip(ordered_phrases, categorical_distances)}
  groups_appearances = {cat: collections.Counter({}) for cat in categories}
  groups_distance = {cat: collections.Counter({}) for cat in categories}
  for phrase, (cat, dist) in phrase2cat.items():
      groups_appearances[cat][phrase] = data.container.appearances[phrase]
      groups_distance[cat][phrase] = 1 - dist


  categorical_weights = np.exp( - (categorical_distances - categorical_distances.min(axis=1)[:, np.newaxis]))
  ordered_appearances = np.array([[data.container.pos_appearances[phrase] for phrase in ordered_phrases],
                                 [data.container.neg_appearances[phrase] for phrase in ordered_phrases]]).T
  category_scores = ((categorical_weights[:, :, np.newaxis] * ordered_appearances[:, np.newaxis]).sum(axis=0) /
                     categorical_weights.sum(axis=0)[:, np.newaxis])

  for category, score in zip(categories, category_scores):
    print(category, score)

  return category_scores


for hotel_dir, data in zip(hotel_files, all_data):
  category_scores = main(data)
  print("\n\n")
  np.save("{}/{}_cat_appearances.npy".format(directories.trip_advisor, hotel_dir), category_scores)
