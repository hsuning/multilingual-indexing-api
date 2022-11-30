from sklearn.metrics.pairwise import cosine_similarity

try:
    import tensorflow.compat.v2 as tf
except:
    import tensorflow as tf

import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import pandas as pd
import numpy as np

class Indexing:
    """
    Wrap all data into one class
    
    -------------------------
    Args:
        url (str): The webpage that we want to extract information

    Returns:
        sub_urls (set): a set (unique list) of urls found in the given webpage
        title: the question extracted from the given webpage
        content: the answer to the question
    """

    def __init__(self, preprocessor, encoder, closest_matches_df):
        # Store loaded modules to avoid reloading
        self.preprocessor = preprocessor
        self.encoder = encoder

        # All questions in closest_matches files
        self.closest_matches_df = closest_matches_df
        self.questions = list(set(closest_matches_df["question"].values))
        self.questions_embeds = None

    def normalization(self, embeds):
        """
        Use l2 normalization to embeddings
        -------------------------
        Args:
            embeds (vector): embedding (high-dimensional) vectors produced by encoder

        Returns:
            norms_embeds (vector): normalized embedding (high-dimensional) vectors
        """
        norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
        return embeds / norms

    def embeddings(self, sentences):
        """
        Encode raw sentences into embedding (high-dimensional) vectors
        -------------------------
        Args:
            embeds (vector): embedding (high-dimensional) vectors

        Returns:
            norms_embeds (vector): normalized embedding (high-dimensional) vectors
        """
        with tf.device('/cpu:0'):
            sentences_embeds = tf.constant(sentences)
            sentences_embeds = self.encoder(
                self.preprocessor(sentences_embeds))["default"]
            # For semantic similarity tasks, apply l2 normalization to embeddings
            sentences_embeds = self.normalization(sentences_embeds)
        return sentences_embeds

    def calculate_similarity(self, embeddings_1, embeddings_2, labels_1,
                             labels_2):
        """
        Calculate the similarity using arccos based text similarity
        of two high-dimensional vectors
        -------------------------
        Args:
            embeddings_1 (vector): embeddings produced by encoder
            embeddings_2: embeddings produced by encoder
            labels_1: texts in used for embeddings_1
            labels_2: texts in used for embeddings_2

        Returns:
            df (dataframe): a pandas dataframe with three columns: 
            (texts in embeddings_1, texts in embeddings_2, similarity between two texts)
        """
        assert len(embeddings_1) == len(labels_1)
        assert len(embeddings_2) == len(labels_2)

        # arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
        sim = 1 - np.arccos(cosine_similarity(embeddings_1,
                                              embeddings_2)) / np.pi

        embeddings_1_col, embeddings_2_col, sim_col = [], [], []
        for i in range(len(embeddings_1)):
            for j in range(len(embeddings_2)):
                embeddings_1_col.append(labels_1[i])
                embeddings_2_col.append(labels_2[j])
                sim_col.append(sim[i][j])
        df = pd.DataFrame(zip(embeddings_1_col, embeddings_2_col, sim_col),
                          columns=['query', 'question', 'sim'])

        df = df.fillna(1)
        return df

    def get_top_n_faqs(self, query, top_n):
        """
        Get top N closest questions based on input text
        -------------------------
        Args:
            query (str): the input text we want to classify against
            top_n (int): the number of results we want to get

        Returns:
            res (dict): a dictionary like 
            {"0":{"question":"Deposit fee","Ranking":1.0,"FAQ_id":132,"locale":"en","market":"en-de"},
            "1":{"question":"Deposit fee","Ranking":1.0,"FAQ_id":132,"locale":"en","market":"en-it"}
        """
        query = [query]

        query_embeds = self.embeddings(query)

        res = self.calculate_similarity(query_embeds, self.questions_embeds,
                                        query, self.questions).nlargest(
                                            top_n, ['sim'])

        res['Ranking'] = res['sim'].rank(ascending=False)
        res = res.merge(self.closest_matches_df, how='left',
                        on='question').drop(['query', 'sim'], axis=1)
        res = res[:top_n].drop('answer', axis=1).T.to_dict()
        return res