import re
import langdetect
from utils import contractions
from typing import Optional
# TODO: Add docstrings


def expand_contractions(text, contraction_mapping=contractions.CONTRACTION_MAP):
  contractions_pattern = re.compile('({})'.format('|'.join(
      contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)

  def expand_match(contraction):
    match = contraction.group(0)
    first_char = match[0]
    if contraction_mapping.get(match):
      expanded_contraction = contraction_mapping.get(match)
    else:
      expanded_contraction = contraction_mapping.get(match.lower())
    expanded_contraction = first_char+expanded_contraction[1:]
    return expanded_contraction

  expanded_text = contractions_pattern.sub(expand_match, text)
  expanded_text = re.sub("'s", "", expanded_text)
  expanded_text = re.sub("'", "", expanded_text)
  return expanded_text


def remove_special_characters(text: str, remove_digits: bool = False) -> str:
  if remove_digits:
    pattern = r"[^.a-zA-z\s]"
  else:
    pattern = r"[^a-zA-z0-9.!?\s]"

  # Substitute all special characters with spaces
  text = re.sub(pattern, " ", text)
  # Substitute any white space character with a single space
  text = " ".join(text.split())
  return text


def find_language(text: str) -> str:
  try:
    language = langdetect.detect(text)
  except:
    print("Failed to identify language of:", text)
    return "<UNK>"
  return language


def is_english(text: str) -> str:
  return find_language(text) == "en"


def preprocessing_pipeline(text: str, check_language: bool = True
                           ) -> Optional[str]:
  """This does the following preprocessing pipeline:

      * Detect language and whether review is good (eg. more than five characters).
      * Expand contractions and remove 's from names.
      * Remove special characters, spaces, newlines, etc. Fullstops are left at this point to distinguish sentences.
      MOVED * Use `neuralcoref` to substitute pronouns with original names.
      MOVED * Replaces person names with the word `Host`

    If this returns `None` then we ignore the review.
  """
  if check_language:
    try:
      language = langdetect.detect(text)
    except:
      print("Failed to identify language of:", text)
      return None
    if language != "en":
      return None

  ptext = expand_contractions(text)
  ptext = remove_special_characters(ptext)
  return ptext