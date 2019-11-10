import os
import pandas as pd
import time
from utils import custom_preprocessing, dependencies

n_message = 200
use_neuralcoref = False
data_dir = "/home/stavros/DATA/TripAdvisorReviews"
hotel_name = "the_kresten_royal_villas_1748reviews"


reviews = pd.read_csv(os.path.join(data_dir, "{}.csv".format(hotel_name)))
reviews = reviews[pd.notnull(reviews["text"])]
print("Loaded {} reviews from {}".format(len(reviews), hotel_name))


n_reviews = len(reviews)
new_columns = list(reviews.columns) + ["processed_text", "lemmatized_text", "aspects"]
processed_data = pd.DataFrame(index=range(n_reviews), columns=new_columns)


start_time = time.time()
i, ic = 0, 0
for i in range(n_reviews):
  data_row = reviews.iloc[i]
  try:
    processed_review = custom_preprocessing.preprocessing_pipeline(data_row["text"],
                                                                   check_language=False,
                                                                   use_neuralcoref=use_neuralcoref,
                                                                   replace_host=False)

    if processed_review is not None:
      processed_data.iloc[ic] = data_row
      processed_data.iloc[ic]["processed_comments"] = processed_review
      aspects, lemmatized_review = dependencies.feature_sentiment(processed_review, lemmatize_text=True)
      processed_data.iloc[ic]["lemmatized_comments"] = lemmatized_review
      processed_data.iloc[ic]["aspects"] = aspects
      ic += 1

      if ic % n_message == 0:
        print("{} / {} processed. - time: {}".format(
            ic + 1, n_reviews, time.time() - start_time))

  except:
    print("Review {} preprocessing failed.".format(i))

# Save to pickle
if use_neuralcoref:
  save_dir = os.path.join(data_dir, "{}_with_aspects.pkl".format(hotel_name))
else:
  save_dir = os.path.join(data_dir, "{}_with_aspects_nocoref.pkl".format(hotel_name))
processed_data.to_pickle(save_dir)