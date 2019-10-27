"""Text preprocessing utilities from Sarkar's guide.

A Practitioner's Guide to Natural Language Processing (Part I)
— Processing & Understanding Text
@medium (Towards Data Science)

Methods in this module are copy & pasted from the above guide.
I turned this into a class for better manipulation of the imported libraries
and avoid loading things when we don't need them (to avoid errors!).
"""
import inspect
import re
import unicodedata
from utils import contractions


class CorpusNormalizer:

    def __init__(self, html_stripping=False, contraction_expansion=True,
                 accented_char_removal=True, text_lower_case=True,
                 text_lemmatization=False, special_char_removal=True,
                 stopword_removal=False, remove_digits=False):
        if html_stripping:
            import bs4
            self.BeautifulSoup = bs4.BeautifulSoup

        if text_lemmatization:
            import spacy
            # I changed spacy's model from 'en_core' to 'en_core_web_sm'
            self.nlp = spacy.load('en_core_web_sm', parse=True, tag=True,
                                  entity=True)
        if stopword_removal:
            import nltk
            from nltk.tokenize import toktok
            self.stopword_list = nltk.corpus.stopwords.words('english')
            self.stopword_list.remove('no')
            self.stopword_list.remove('not')
            self.tokenizer = toktok.ToktokTokenizer()

        # Make given arguments attributes so that user doesn't have to give
        # them again in `__call__`.
        self.argnames = set(inspect.getfullargspec(self.__init__).args)
        self.argnames.remove("self")
        for arg in self.argnames:
            setattr(self, arg, locals()[arg])

    def strip_html_tags(self, text):
        soup = self.BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    @staticmethod
    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    @staticmethod
    def expand_contractions(text, contraction_mapping=contractions.CONTRACTION_MAP):

        contractions_pattern = re.compile('({})'.format('|'.join(
            contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    @staticmethod
    def remove_special_characters(text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    def lemmatize_text(self, text):
        text = self.nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text
                         for word in text])
        return text

    def remove_stopwords(self, text, is_lower_case=False):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens
                               if token not in self.stopword_list]
        else:
            filtered_tokens = [token for token in tokens
                               if token.lower() not in self.stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def __call__(self, corpus, n_message=None):
        normalized_corpus = []
        # normalize each document in the corpus
        for i, doc in enumerate(corpus):
            # strip HTML
            # (disabled `html_stripping` as we are not using scrapped data))
            #if html_stripping:
            #    doc = strip_html_tags(doc)
            # remove accented characters
            if self.accented_char_removal:
                doc = self.remove_accented_chars(doc)
            # expand contractions
            if self.contraction_expansion:
                doc = self.expand_contractions(doc)
            # lowercase the text
            if self.text_lower_case:
                doc = doc.lower()
            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
            # lemmatize text
            if self.text_lemmatization:
                doc = self.lemmatize_text(doc)
            # remove special characters and\or digits
            if self.special_char_removal:
                # insert spaces between special characters to isolate them
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = self.remove_special_characters(
                    doc, remove_digits=self.remove_digits)
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            # remove stopwords
            if self.stopword_removal:
                doc = self.remove_stopwords(
                    doc, is_lower_case=self.text_lower_case)

            normalized_corpus.append(doc)
            if n_message is not None and i % n_message == 0:
                print("{} / {} done.".format(i + 1, len(corpus)))

        return normalized_corpus

    def __str__(self):
        enabled_functionalities = [arg for arg in self.argnames
                                   if getattr(self, arg)]
        return "\n".join(enabled_functionalities)