import numpy as np
from time import time
from scorers import Scorers
from regularizers import Regularizers
from cutils import update_triplet_counts


class TopicsModel():
    """ Probabilistic model for finding latent topics distribution in a set of documents.

    Attributes
    ----------
        n_topics: integer
            The number of latent topics to find.
        n_words: ineger
            The number of words in dictionary.
        n_documents: integer
            The size of a set of documents.
        dictionary: list
            The list of words to use during training.
        scorers: Scorers

    """
    EPS = 1e-16

    def __init__(self, n_topics: int, dictionary: list, scorers: list = None, random_state=None,
                 word_topic_reg: list = None, topic_document_reg: list = None):

        self.n_topics = n_topics
        self.n_words = len(dictionary)
        self.n_documents = None
        self.batch_size = None
        self.dictionary = dictionary

        self.word_topic_dist = None
        self.topic_document_dist = None

        self._seed = random_state
        self.scorers = Scorers(scorers)
        self.word_topic_reg = Regularizers(word_topic_reg)
        self.topic_document_reg = Regularizers(topic_document_reg)

    def _get_start_topics_dists(self):
        """ Returns initial approximation of topics distribution.

        The topics distribution consis of two matrices:
            - words per topic distribution matrix
            - topics per document distribution matrix
        """
        # Set the random state for reproducibility of results.
        state = np.random.RandomState(self._seed)

        word_topic_dist = state.uniform(0, 1, size=(self.n_words, self.n_topics))
        word_topic_dist = np.divide(
            word_topic_dist,
            word_topic_dist.sum(axis=0) + TopicsModel.EPS)

        topic_document_dist = state.uniform(0, 1, size=(self.n_topics, self.n_documents))
        topic_document_dist = np.divide(
            topic_document_dist,
            topic_document_dist.sum(axis=1)[:, None] + TopicsModel.EPS)

        return word_topic_dist, topic_document_dist

    def fit(self, term_doc_matrix, batch_size=2000, tol=1e-6, max_iter=None, verbose=0):
        """ Runs the EM-algorithm to find latent topics distribution.

        Arguments
        ---------
            term_doc_matrix: csr_matrix
                The sparse matrix of terms and documents.
            batch_size: integer, optional (default=2000)
                The number of rows in term_doc_matrix to convert to dense format per iteration.
            tol: float, optional (default=1e-6)
                The tolerance for EM-algorithm convergence.
                Infinity norm for difference of previous and current distribution matrices.
            max_iter: integer, optional (default=None)
                The maximal number of EM iterations.
            verbose: integer, optional (default=0)
                The level of loggin. '0' means no logs.
        """
        self.n_documents = term_doc_matrix.shape[0]
        self.batch_size = batch_size

        if max_iter is None:
            max_iter = np.inf

        if self.word_topic_dist is None and self.topic_document_dist is None:
            self.word_topic_dist, self.topic_document_dist = self._get_start_topics_dists()
        word_topic_error, topic_document_error = np.inf, np.inf
        if verbose > 0:
            print(" word/topic error  |  topic/document error ")
            print(" ------------------------------------------")

        n_wt = np.zeros(shape=(self.n_words, self.n_topics), dtype=np.float64)
        n_td = np.zeros(shape=(self.n_topics, self.n_documents), dtype=np.float64)

        start = time()
        it = 0
        while (word_topic_error > tol or topic_document_error > tol) and it < max_iter:
            n_wt.fill(0.)
            n_td.fill(0.)

            for i in np.arange(0, self.n_documents, batch_size):
                term_doc_matrix_batch = np.asarray(term_doc_matrix[i:i + batch_size].todense())
                # The cython function for fast update.
                update_triplet_counts(self.word_topic_dist, self.topic_document_dist[:, i:i + batch_size],
                                      term_doc_matrix_batch, n_wt, n_td, i, i + term_doc_matrix_batch.shape[0],
                                      self.n_topics)

            if not self.word_topic_reg.empty():
                word_topic_dist_new = self.word_topic_reg.apply(n_wt)
            else:
                word_topic_dist_new = n_wt / (n_wt.sum(axis=0) + TopicsModel.EPS)

            if not self.topic_document_reg.empty():
                topic_document_dist_new = self.topic_document_reg.apply(n_td)
            else:
                topic_document_dist_new = n_td / (n_td.sum(axis=0) + TopicsModel.EPS)

            word_topic_error = np.abs(word_topic_dist_new - self.word_topic_dist).max()
            topic_document_error = np.abs(topic_document_dist_new - self.topic_document_dist).max()
            if verbose > 1:
                print("      {:.6f}     |      {:.6f}      ".format(word_topic_error, topic_document_error))

            self.word_topic_dist = word_topic_dist_new
            self.topic_document_dist = topic_document_dist_new
            it += 1

        if verbose > 0:
            print("      {:.6f}     |      {:.6f}      ".format(word_topic_error, topic_document_error))
            print(f"\nTotal time: {time() - start:.8f}s")

    def score(self, term_doc_matrix):
        return self.scorers.score(term_doc_matrix, self)

    def print_topics(self, words_in_topic=5):
        vocab = np.array(self.dictionary)
        for t in range(self.n_topics):
            probas = self.word_topic_dist.T[t]
            probas_id = np.argsort(probas)[::-1][:words_in_topic]
            words = vocab[probas_id]
            weights = probas[probas_id]
            print(f"Topic {t + 1}: " + " + ".join([f"{weights[i]:.4f}*'{words[i]}'" for i in range(words_in_topic)]))
