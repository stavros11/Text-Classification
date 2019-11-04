import os
import pandas as pd
import time
from utils import custom_preprocessing, dependencies


area = "athens"
n_message = 400
n_save = 5000


data_dir = "/home/stavros/DATA/AirbnbReviews"
area_dir = os.path.join(data_dir, area)
reviews = pd.read_csv(os.path.join(area_dir, "reviews.csv.gz"))
reviews = reviews[pd.notnull(reviews["listing_id"])]
reviews = reviews[pd.notnull(reviews["comments"])]

print("Loaded {} reviews found for {}".format(len(reviews), area))

# Preprocess the reviews of all listings that have more than 300 reviews
valid_listings = set(ids for ids, n in reviews["listing_id"].value_counts().items() if n > 300)
valid_reviews = reviews[reviews["listing_id"].map(lambda x: x in valid_listings)]
n_samples = len(valid_reviews)
n_listings = len(valid_listings)

print("{} reviews found for {} valid listings".format(n_samples, n_listings))
print("Saving checkpoints every {} points".format(n_save))

# Function that create checkpoint names
def get_pkl_name(n: int) -> str:
  data = "listings{}_samples{}".format(n_listings, n)
  return os.path.join(area_dir, "reviews_with_aspects_{}.pkl".format(data))


sampled_columns = list(valid_reviews.columns) + ["processed_comments", "lemmatized_comments", "aspects"]
sampled_data = pd.DataFrame(index=range(n_samples), columns=sampled_columns)


start_time = time.time()
i, ic = 0, 0
while i < n_samples:
  data_row = valid_reviews.iloc[i]
  i += 1
  try:
    processed_review = custom_preprocessing.preprocessing_pipeline(data_row["comments"])

    if processed_review is not None:
      sampled_data.iloc[ic] = data_row
      sampled_data.iloc[ic]["processed_comments"] = processed_review
      aspects, lemmatized_review = dependencies.feature_sentiment(processed_review, lemmatize_text=True)
      sampled_data.iloc[ic]["lemmatized_comments"] = lemmatized_review
      sampled_data.iloc[ic]["aspects"] = aspects
      ic += 1

      if ic % n_message == 0:
        print("{} / {} found. - time: {}".format(ic + 1, n_samples, time.time() - start_time))
      if ic % n_save == 0:
        sampled_data.to_pickle(get_pkl_name(ic))
        print("Saved checkpoint with {} samples".format(ic))
  except:
    pass

# Save to pickle
sampled_data.to_pickle(get_pkl_name(n_samples))