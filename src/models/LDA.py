from typing import List, Tuple

import spacy
from gensim import corpora, models
from gensim.corpora import Dictionary


class LDA:
    def __init__(self, data: list[str], num_topics=5, language='nb_core_news_md'):
        self.data = data
        self.nlp = spacy.load(language)
        self.model = self._build_LDA_model(num_topics)

    def _preprocess(self) -> list[list[str]]:
        """
        Clean and extract important words and noun phrases.
        Args:
            text (str): The input document to be preprocessed.
        Returns:
            List[str]: A list of words and noun phrases.

        """
        preprocessed = []

        for i, text in enumerate(self.data):
            if not isinstance(text, str):
                continue

            doc = self.nlp(text)

            cleaned_noun_chunks = []
            for chunk in doc.noun_chunks:
                clean_chunk = ' '.join([token.lemma_.lower() for token in chunk if not token.is_stop and token.is_alpha])
                if clean_chunk:
                    cleaned_noun_chunks.append(clean_chunk)

            tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
            combined = list(set(tokens + cleaned_noun_chunks))
            preprocessed.append(combined)

        return preprocessed

    def _build_corpus(self) -> tuple[list[list[tuple[int, int]]], Dictionary]:
        preprocessed_docs = self._preprocess()
        dictionary = corpora.Dictionary(preprocessed_docs)

        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]
        return corpus, dictionary

    def _build_LDA_model(self, num_topics):
        corpus, dictionary = self._build_corpus()
        lda_model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        return lda_model
