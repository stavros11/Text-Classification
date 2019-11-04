import os
import pandas as pd
from utils import dependencies


area = "south_aegean"
full_filename = "reviews_with_aspects_Santorini_100266samples.pkl"
n_message = 500

filename, filetype = full_filename.split(".")
data_dir = "/home/stavros/DATA/AirbnbReviews"
area_dir = os.path.join(data_dir, area)
if filetype == "pkl":
  clean_data = pd.read_pickle(os.path.join(area_dir, full_filename))
else:
  raise NotImplementedError


# Print messages to know what the script is doing
n_samples = len(clean_data)
print("{} reviews read from {}".format(n_samples, full_filename))

counter = dependencies.FeatureSentimentCounter(n_message)
sentiments = clean_data["processed_comments"].map(counter.feature_sentiment)

# Save to pickle
clean_data["aspects_sentiment"] = sentiments
clean_data.to_pickle(os.path.join(area_dir, "{}_sentiment.pkl".format(filename)))