import os
import numpy as np
import pandas as pd
import time
from utils import custom_preprocessing, dependencies


n_samples = 200000
area = "nyc"
island = None
n_message = 400
n_save = 5000

data_dir = "/home/stavros/DATA/AirbnbReviews"
area_dir = os.path.join(data_dir, area)

if island is None:
  # Either load the downloaded csv
  data = pd.read_csv(os.path.join(area_dir, "reviews.csv.gz"))
  clean_data = data[pd.notnull(data["comments"])]
  get_pkl_name = lambda n: os.path.join(area_dir, "reviews_with_aspects_{}samples.pkl".format(n))

else:
  # or some other file we created (eg. with island locations for southern agean)
  clean_data = pd.read_pickle(os.path.join(area_dir, "reviews_with_location.pkl"))
  # Keep only reviews from the given island
  clean_data = clean_data[clean_data.location == island]
  print("Keeping reviews from {} only.".format(island))
  get_pkl_name = lambda n: os.path.join(area_dir, "reviews_with_aspects_{}_{}samples.pkl".format(island, n))

ids = np.arange(len(clean_data))
np.random.shuffle(ids)
if n_samples is None or n_samples > len(clean_data):
  n_samples = len(clean_data)

# Print messages to know what the script is doing
print("{} reviews found for {}".format(len(clean_data), area))
print("Target number of samples:", n_samples)
print("Saving checkpoints every {} samples.".format(n_save))

sampled_columns = list(clean_data.columns) + ["processed_comments", "aspects"]
sampled_data = pd.DataFrame(index=range(n_samples), columns=sampled_columns)
# Save shuffled ids in case you want to restart for the rest of the reviews later
if island is None:
  np.save(os.path.join(area_dir, "review_aspact_finding_random_ids.npy"), ids)
else:
  np.save(os.path.join(area_dir, "review_aspact_finding_random_ids_{}.npy".format(island)), ids)

start_time = time.time()
i, ic = 0, 0
while ic < n_samples and i < len(ids):
  data_row = clean_data.iloc[ids[i]]
  i += 1
  try:
    processed_review = custom_preprocessing.preprocessing_pipeline(data_row["comments"])

    if processed_review is not None:
      sampled_data.iloc[ic] = data_row
      sampled_data.iloc[ic]["processed_comments"] = processed_review
      sampled_data.iloc[ic]["aspects"] = dependencies.find_features(processed_review)
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