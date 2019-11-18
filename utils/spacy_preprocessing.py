import re
import spacy
import time
from spacy import tokens
from typing import Iterable, List


def replace_host(docs: Iterable[tokens.Doc]) -> List[str]:
  """Replaces PERSON entities with the word 'Host'."""
  texts = []
  for doc in docs:
    texts.append(doc.text)
    names = {token.text for token in doc.ents if token.label_ == "PERSON"}
    for name in names:
      texts[-1] = re.sub(name, "Host", texts[-1])
  return texts


def lemmatize(docs: Iterable[tokens.Doc]) -> List[str]:
  texts = []
  start_time = time.time()

  for doc in docs:
    text = " ".join([token.lemma_ if token.lemma_ != '-PRON-' else token.text
                     for token in doc])
    # Leave only letter characters
    text = re.sub("[^a-zA-z\s]", " ", text)
    # Substitute any white space character with a single space
    text = " ".join(text.split())
    # Make lower case
    texts.append(text.lower())

  print("\nLemmatized {} reviews.".format(len(texts)))
  print(time.time() - start_time)
  return texts


def apply_spacy(texts: Iterable[str], parse=True, tag=True, entity=True
                ) -> Iterable[tokens.Doc]:
  nlp = spacy.load('en_core_web_sm', parse=parse, tag=tag, entity=entity)

  start_time = time.time()
  docs = list(nlp.pipe(texts))
  print("\nApplied spacy on {} reviews.".format(len(docs)))
  print(time.time() - start_time)

  return docs


def apply_neuralcoref(texts: Iterable[str]) -> List[str]:
  import neuralcoref
  nlp = spacy.load('en_core_web_sm', parse=False, tag=False, entity=False)
  nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab), name='neuralcoref')

  start_time = time.time()
  docs = nlp.pipe(texts)
  resolved = [doc._.coref_resolved for doc in docs]
  print("\nApplied neuralcoref on {} reviews.".format(len(texts)))
  print(time.time() - start_time)
  return resolved