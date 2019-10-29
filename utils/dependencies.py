import collections
import os
import spacy
from typing import Set


def load_words(lexicon_dir: str) -> Set[str]:
    """Loads opinion word from txt file to set."""
    file = open(os.path.join(lexicon_dir), encoding="ISO-8859-1")
    return set(line.strip() for line in file.readlines())

_NLP = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
_LEXICON_DIR = "/home/stavros/GitHub/Text-Classification/opinion-lexicon"
_POS_WORDS = load_words(os.path.join(_LEXICON_DIR, "pos_words.txt"))
_NEG_WORDS = load_words(os.path.join(_LEXICON_DIR, "neg_words.txt"))
_OPINION_WORDS = _POS_WORDS | _NEG_WORDS


def feature_sentiment(sentence: str) -> collections.Counter:
    """Finds feature words and the corresponding sentiment.

    Example use:
      feature_sentiment("The pillows were very uncomfortable. "
                        "Location was great")
      Returns: Counter({location: 1, pillows: -1})

    Args:
      sentence: A review text. Can have more than one sentence, for example
        can be a full review comment.

    Returns:
      A counter where keys are the features and the values are the
        corresponding sentiment scores.
    """
    sent_dict = collections.Counter()
    sentence = _NLP(sentence)
    debug = 0
    for token in sentence:
        # check if the word is an opinion word, then assign sentiment
        if token.text in _OPINION_WORDS:
            sentiment = 1 if token.text in _POS_WORDS else -1
            # if target is an adverb modifier (i.e. pretty, highly, etc.)
            # but happens to be an opinion word, ignore and pass
            if (token.dep_ == "advmod"):
                continue
            elif (token.dep_ == "amod"):
                sent_dict[token.head.text] += sentiment
            # for opinion words that are adjectives, adverbs, verbs...
            else:
                for child in token.children:
                    # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                    # This could be better updated for modifiers that either positively or negatively emphasize
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in _OPINION_WORDS):
                        sentiment *= 1.5
                    # check for negation words and flip the sign of sentiment
                    if child.dep_ == "neg":
                        sentiment *= -1
                for child in token.children:
                    # if verb, check if there's a direct object
                    if (token.pos_ == "VERB") & (child.dep_ == "dobj"):
                        sent_dict[child.text] += sentiment
                        # check for conjugates (a AND b), then add both to dictionary
                        subchildren = []
                        conj = 0
                        for subchild in child.children:
                            if subchild.text == "and":
                                conj=1
                            if (conj == 1) and (subchild.text != "and"):
                                subchildren.append(subchild.text)
                                conj = 0
                        for subchild in subchildren:
                            sent_dict[subchild] += sentiment

                # check for negation
                for child in token.head.children:
                    noun = ""
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in _OPINION_WORDS):
                        sentiment *= 1.5
                    # check for negation words and flip the sign of sentiment
                    if (child.dep_ == "neg"):
                        sentiment *= -1

                # check for nouns
                for child in token.head.children:
                    noun = ""
                    if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                        noun = child.text
                        # Check for compound nouns
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                noun = subchild.text + " " + noun
                        sent_dict[noun] += sentiment
                    debug += 1
    return sent_dict


def find_features(sentence: str) -> collections.Counter:
    """Same as above but only counts features without caring about sentiment."""
    sent_dict = collections.Counter()
    sentence = _NLP(sentence)
    debug = 0
    for token in sentence:
        # check if the word is an opinion word, then assign sentiment
        if token.text in _OPINION_WORDS:
            # if target is an adverb modifier (i.e. pretty, highly, etc.)
            # but happens to be an opinion word, ignore and pass
            if (token.dep_ == "advmod"):
                continue
            elif (token.dep_ == "amod"):
                sent_dict[token.head.text] += 1
            # for opinion words that are adjectives, adverbs, verbs...
            else:
                for child in token.children:
                    # if verb, check if there's a direct object
                    if (token.pos_ == "VERB") & (child.dep_ == "dobj"):
                        sent_dict[child.text] += 1
                        # check for conjugates (a AND b), then add both to dictionary
                        subchildren = []
                        conj = 0
                        for subchild in child.children:
                            if subchild.text == "and":
                                conj=1
                            if (conj == 1) and (subchild.text != "and"):
                                subchildren.append(subchild.text)
                                conj = 0
                        for subchild in subchildren:
                            sent_dict[subchild] += 1

                # check for nouns
                for child in token.head.children:
                    noun = ""
                    if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                        noun = child.text
                        # Check for compound nouns
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                noun = subchild.text + " " + noun
                        sent_dict[noun] += 1
                    debug += 1
    return sent_dict