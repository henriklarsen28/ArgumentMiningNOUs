import spacy
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt


class LDA:
    def __init__(self, data: list[str], no_below=15, no_above=0.5, language='nb_core_news_md'):
        self.data = data
        self.nlp = spacy.load(language)
        self.preprocessed_data = self._preprocess()
        self.corpus, self.dictionary = self._build_corpus(no_below=no_below, no_above=no_above)

    def _preprocess(self) -> list[list[str]]:
        """
        Clean and extract important words and noun phrases.
        """
        preprocessed = []

        for i, text in enumerate(self.data):
            if not isinstance(text, str):
                raise ValueError('Non-string value, Please Check data fields')

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

    def build_LDA_model(self, num_topics, passes):
        lda_model = models.LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes)
        return lda_model

    def predict_topics(self, model, relevancy=False):
        predictions = []
        for i, bow in enumerate(self.corpus):
            topic_distribution = model.get_document_topics(bow)

            sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
            top_topic, top_prob = sorted_topics[0]  # Top topic and its probability

            prediction = {'topic': top_topic, 'confidence': round(top_prob, 2)}
            predictions.append(prediction)

            if relevancy:
                top_topic_words = model.show_topic(top_topic, topn=1000)
                top_words = [word for word, prob in top_topic_words]
                preprocessed_words_set = set(self.preprocessed_data[i])
                relevant_words = [word for word in top_words if word in preprocessed_words_set]
                prediction['relevant_words'] = relevant_words

        return predictions

    def extract_arguments(self, document):
        pass

    def plot_coherence(self, topic_interval=(2, 12), passes=10, savefig=None):
        coherence_values = []
        model_list = []
        topic_range = range(topic_interval[0], topic_interval[1] + 1)

        for num_topics in topic_range:
            model = self.build_LDA_model(num_topics=num_topics, passes=passes)
            model_list.append(model)
            coherence_model = CoherenceModel(model=model, texts=self.preprocessed_data, dictionary=self.dictionary,
                                             coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())

        # Plotting the coherence values
        plt.figure(figsize=(12, 6))
        plt.plot(topic_range, coherence_values, label=f'Passes: {passes}')
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.title("Coherence Scores for LDA Models")
        plt.xticks(topic_range)
        plt.legend()

        if savefig:
            plt.savefig(f'../../plots/{savefig}.png')
        plt.show()


def plot_topic_distribution(dataframe, savefig=None):
    dataframe['topic'] = dataframe['topic_predictions'].apply(lambda x: x['topic'])
    grouped = dataframe.groupby(['topic', 'actor_label']).size().unstack(fill_value=0)

    class_totals = grouped.sum()  # Frequency of each class
    grouped_percentage = grouped.div(class_totals) * 100

    # Plotting the stacked bar chart with percentages
    ax = grouped_percentage.plot(kind='bar', stacked=True, figsize=(10, 7))
    plt.title('Percentage Distribution of Topics Segmented by Class Relative to Class Total')
    plt.xlabel('Topic')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.legend(title='Class')

    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')

    if savefig:
        plt.savefig(f'../../plots/{savefig}.png')

    plt.show()



