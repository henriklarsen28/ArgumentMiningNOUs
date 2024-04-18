import pandas as pd
import spacy
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.data_processing.text_formatting import cleanup_whitespaces


class LDA:
    def __init__(self, data: list[str], language, no_below=15, no_above=0.5, filter_extremes=True):
        """
        Initializes the LDA class with the provided data and configuration settings.

        Parameters:
            data (list[str]): List of documents as strings.
            no_below (int): Minimum number of documents a word must appear in to be kept.
            no_above (float): Maximum proportion of documents a word can appear in to be kept.
            language (str): The spaCy model to use for text processing.
        """
        self.data = data
        self.language = language
        self.nlp = spacy.load(language)
        self.preprocessed_data = self._preprocess()
        self.corpus, self.dictionary = self._build_corpus(no_below=no_below, no_above=no_above,
                                                          filter_extremes=filter_extremes)

    def _preprocess(self) -> list[list[str]]:
        """
        Cleans and extracts important words and noun phrases from the initial data.

        Returns:
            list[list[str]]: A list of documents where each document is a list of preprocessed tokens.
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

    def _build_corpus(self, no_below=15, no_above=0.5, filter_extremes=True):
        """
        Builds the corpus and dictionary from the preprocessed data using specified filtering parameters.

        Parameters:
            no_below (int): Minimum number of documents a word must appear in.
            no_above (float): Maximum proportion of documents a word can appear in.

        Returns:
            tuple: Contains the corpus (list of lists of (int, int) tuples) and the dictionary (gensim Dictionary).
        """
        dictionary = corpora.Dictionary(self.preprocessed_data)

        if filter_extremes:
            dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=100000)

        if not dictionary:
            raise ValueError("Empty dictionary generated from data")

        corpus = [dictionary.doc2bow(doc) for doc in self.preprocessed_data]

        if not any(corpus):
            raise ValueError("Empty corpus generated from dictionary")
        return corpus, dictionary

    def build_LDA_model(self, num_topics, passes):
        """
        Builds and returns an LDA model with the specified number of topics and passes.

        Parameters:
            num_topics (int): Number of topics for the LDA model.
            passes (int): Number of passes through the corpus during training.

        Returns:
            gensim.models.LdaModel: The trained LDA model.
        """
        lda_model = models.LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes)
        return lda_model

    def predict_topics(self, model, relevancy=False):
        """
        Predicts the most relevant topics for the documents in the corpus using the specified model.

        Parameters:
            model (gensim.models.LdaModel): The LDA model used for prediction.
            relevancy (bool): If True, also returns relevant words from the topics for each document.

        Returns:
            list[dict]: List of dictionaries containing the topic and confidence of predictions, and optionally relevant words.
        """
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

    def calculate_lda_model_coherences(self, topic_interval=(2, 12), passes=10):
        """
        Calculates the coherence scores for a range of topic numbers to evaluate LDA models.

        Parameters:
            topic_interval (tuple[int, int]): Start and end of the range of topics to test.
            passes (int): Number of passes through the corpus for each model.

        Returns:
            tuple: Contains the list of coherence values, the list of LDA models, and the topic range.
        """
        coherence_values = []
        model_list = []
        topic_range = range(topic_interval[0], topic_interval[1] + 1)

        for num_topics in tqdm(topic_range, 'Building LDA-models'):
            model = self.build_LDA_model(num_topics=num_topics, passes=passes)
            model_list.append(model)
            coherence_model = CoherenceModel(model=model, texts=self.preprocessed_data, dictionary=self.dictionary,
                                             coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())

        return coherence_values, model_list, topic_range, passes

    def extract_arguments(self, dataframe, model):
        arguments_df = pd.DataFrame(columns=['actor', 'text', 'label'])

        for _, row in tqdm(dataframe.iterrows(), desc="Processing Documents", total=len(dataframe)):
            actor = row['actor']
            document = row['text']
            label = row['label']

            doc = self.nlp(document)
            sentences = [sentence.text for sentence in doc.sents]

            sentence_topics = self.predict_topics(model)

            # Find sequences of sentences with the same topic longer than two
            current_topic = None
            topic_sequence = []
            for sentence, prediction in zip(sentences, sentence_topics):
                if prediction['topic'] == current_topic:
                    topic_sequence.append(sentence)
                else:
                    if len(topic_sequence) > 2:
                        arguments_text = ' '.join(topic_sequence)
                        arguments_df = arguments_df.append({'actor': actor, 'text': arguments_text, 'label': label},
                                                           ignore_index=True)
                    current_topic = prediction['topic']
                    topic_sequence = [sentence]

            # Catch the last sequence in the document
            if len(topic_sequence) > 2:
                arguments_text = ' '.join(topic_sequence)
                arguments_df = arguments_df.append({'actor': actor, 'text': arguments_text, 'label': label},
                                                   ignore_index=True)

        return arguments_df


def plot_coherence_scores(topic_range, coherence_values, passes, savefig=None):
    """
    Plots coherence scores over a range of topic numbers to visualize the performance of LDA models.

    Parameters:
        topic_range (range): The range of topics over which coherence was computed.
        coherence_values (list[float]): The list of coherence scores corresponding to each topic in topic_range.
        passes (int): Number of passes through the corpus for each model used in coherence calculation.
        savefig (str, optional): Path to save the figure to. If None, the figure is shown but not saved.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(topic_range, coherence_values, label=f'Passes: {passes}')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title("Coherence Scores for LDA Models")
    plt.xticks(topic_range)
    plt.legend()

    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_topic_distribution(dataframe, savefig=None):
    """
    Plots the distribution of topics in the data, segmented by class.

    Parameters:
        dataframe (pandas.DataFrame): DataFrame containing topic predictions and class labels.
        savefig (str, optional): Filename to save the plot. If None, the plot is not saved.
    """
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
