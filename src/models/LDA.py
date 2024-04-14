from typing import List, Tuple

import spacy
from gensim import corpora, models
from gensim.corpora import Dictionary


class LDA:
    def __init__(self, data: list[str], num_topics=5, no_below=15, no_above=0.5, passes=10, language='nb_core_news_md'):
        self.data = data
        self.preprocessed_data = self._preprocess()
        self.nlp = spacy.load(language)
        self.corpus, self.dictionary = self._build_corpus(no_below=no_below, no_above=no_above)
        self.model = self._build_LDA_model(num_topics, passes)

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
                clean_chunk = ' '.join(
                    [token.lemma_.lower() for token in chunk if not token.is_stop and token.is_alpha])
                if clean_chunk:
                    cleaned_noun_chunks.append(clean_chunk)

            tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
            combined = list(set(tokens + cleaned_noun_chunks))
            preprocessed.append(combined)

        return preprocessed

    def _build_corpus(self, no_below=15, no_above=0.5) -> tuple[list[list[tuple[int, int]]], Dictionary]:
        dictionary = corpora.Dictionary(self.preprocessed_data)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=100000)
        corpus = [dictionary.doc2bow(doc) for doc in self.preprocessed_data]
        return corpus, dictionary

    def _build_LDA_model(self, num_topics, passes):
        lda_model = models.LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes)
        return lda_model

    def predict_topics(self, relevancy=False):
        predictions = []
        for i, bow in enumerate(self.corpus):
            topic_distribution = self.model.get_document_topics(bow)

            sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
            top_topic, top_prob = sorted_topics[0]  # Top topic and its probability

            prediction = {'topic': top_topic, 'confidence': round(top_prob, 2)}
            predictions.append(prediction)

            if relevancy:
                top_topic_words = self.model.show_topic(top_topic, topn=1000)
                top_words = [word for word, prob in top_topic_words]
                preprocessed_words_set = set(self.preprocessed_data[i])
                relevant_words = [word for word in top_words if word in preprocessed_words_set]
                prediction['relevant_words'] = relevant_words

        return predictions
