import os
import argparse
import pandas as pd
from aspects import find_aspects
from utils import directories
from utils import basic_preprocessing, spacy_preprocessing


def load_reviews(hotel_name: str, n_reviews: str) -> pd.DataFrame:
  loadname = "{}_{}reviews.csv".format(hotel_name, n_reviews)
  loadname = os.path.join(directories.trip_advisor, loadname)

  reviews = pd.read_csv(loadname)
  reviews = reviews[pd.notnull(reviews["text"])]
  return reviews


def preprocessing(texts: pd.Series):
  texts = texts.map(basic_preprocessing.expand_contractions)
  texts = texts.map(basic_preprocessing.remove_special_characters)
  print("Basic preprocessing completed on {} reviews.".format(len(texts)))
  return texts


def main(hotel_name: str, n_reviews: int,
         skip_language_check: bool = False,
         apply_basic_preprocessing: bool = False,
         use_neuralcoref: bool = False):
  # List that keeps track which preprocessing options where used so that
  # we log them in the saved pickle title
  # Log the number of reviews right before saving because this changes as
  # some reviews are not valid
  savename = [hotel_name, "{}reviews", "withaspects"]

  reviews = load_reviews(hotel_name, n_reviews)
  n_reviews = len(reviews)
  print("Loaded {} reviews from {}".format(n_reviews, hotel_name))

  if skip_language_check:
    # Keep reviews with more than 2 characters
    valid_reviews = reviews[reviews.text.map(lambda x: len(x)) > 2]
    print("Kept {} reviews with more than 2 characters.".format(
        len(valid_reviews)))
  else:
    valid_reviews = reviews[reviews.text.map(basic_preprocessing.is_english)]
    print("Kept {} english reviews.".format(len(valid_reviews)))

  n_reviews = len(valid_reviews)
  texts = valid_reviews.text

  # Basic preprocessing
  if apply_basic_preprocessing:
    savename.append("basicproc")
    texts = preprocessing(valid_reviews.text)

  # Use neuralcoref
  # TODO: Think of a way to run spacy once for both coref, host name
  # substitution and aspect identification
  if use_neuralcoref:
    savename.append("coref")
    texts = spacy_preprocessing.apply_neuralcoref(texts)

  # Create spacy docs using `nlp.pipe`
  spacy_docs = spacy_preprocessing.apply_spacy(texts)
  # Use docs to find aspects
  aspects = find_aspects.sentiment_aspects(spacy_docs)
  # Lemmatize text after finding aspects
  lemmatized_texts = spacy_preprocessing.lemmatize(spacy_docs)

  # Add columns to the DataFrame
  pd.options.mode.chained_assignment = None
  valid_reviews["processed_text"] = texts
  valid_reviews["aspects"] = aspects
  valid_reviews["lemmatized_text"] = lemmatized_texts

  # Save to pickle
  savename = "_".join(savename).format(n_reviews)
  valid_reviews.to_pickle("{}.pkl".format(
      os.path.join(directories.trip_advisor, savename)))
  print("\nSaved DataFrame with shape {} to {}.".format(
      valid_reviews.shape, savename))

  return valid_reviews


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument("--hotel-name", type=str)
  parser.add_argument("--n-reviews", type=int)
  parser.add_argument("--skip-language-check", action="store_true")
  parser.add_argument("--apply-basic-preprocessing", action="store_true")
  parser.add_argument("--use-neuralcoref", action="store_true")
  #parser.add_argument("--n-message", type=int, default=200)

  args = parser.parse_args()
  main(**vars(args))