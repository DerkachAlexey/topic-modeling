from abc import ABCMeta, abstractmethod

import numpy as np
from cutils import likelihood


class BaseScorer(metaclass=ABCMeta):
    """ Base class for scoring topics model. """

    @staticmethod
    @abstractmethod
    def score(term_doc_matrix, model):
        """ Returns a score of topics model that.

        Arguments
        ---------
            term_doc_matrix: csr_matrix
                The term-document matrix which was used to fit model.
            model: TopicsModel
                The trained topics model.

        Returns
        -------
            The model's score.
        """

    @staticmethod
    @abstractmethod
    def name():
        """ Returns the name of scorer. """


class PerplexityScorer(BaseScorer):

    @staticmethod
    def score(term_doc_matrix, model):
        n_documents = term_doc_matrix.shape[0]
        n_words = term_doc_matrix.shape[1]
        total_likelihood = 0.0
        for i in np.arange(0, n_documents, model.batch_size):
            term_doc_matrix_batch = np.asarray(term_doc_matrix[i:i + model.batch_size].todense())
            batch_doc_size = term_doc_matrix_batch.shape[0]
            topic_document_dist_batch = model.topic_document_dist[:, i:i + batch_doc_size]
            total_likelihood += likelihood(
                term_doc_matrix_batch,
                model.word_topic_dist,
                topic_document_dist_batch,
            )
        return np.exp(-(1. / n_words) * total_likelihood)

    @staticmethod
    def name():
        return "perplexity"


class RelativePerplexityScorer(PerplexityScorer):

    @staticmethod
    def _min_max_perplexity(term_doc_matrix):
        n_words = term_doc_matrix.shape[1]
        p_dw_min = term_doc_matrix.multiply(1. / term_doc_matrix.sum(axis=1))
        p_dw_min.data = np.log(p_dw_min.data)
        p_dw_max = term_doc_matrix.sum(axis=0) / (term_doc_matrix.sum() + 1e-12)
        max_perp = np.exp(-(1. / n_words) * term_doc_matrix.multiply(np.log(p_dw_max)).sum())
        min_perp = np.exp(-(1. / n_words) * term_doc_matrix.multiply(p_dw_min).sum())
        return min_perp, max_perp

    @staticmethod
    def score(term_doc_matrix, model):
        perp = PerplexityScorer.score(term_doc_matrix, model)
        min_perp, max_perp = RelativePerplexityScorer._min_max_perplexity(term_doc_matrix)
        return (perp - min_perp) / (max_perp - min_perp)

    @staticmethod
    def name():
        return "relative_perplexity"


class Scorers():

    def __init__(self, scorers: list = None):
        self._scorers = scorers
        if scorers is None:
            self._scorers = []

    def add(self, scorer):
        self._scorers.append(scorer)

    def score(self, term_doc_matrix, model):
        if not self._scorers:
            raise ValueError("No scorers were added.")

        res = {}
        for scorer in self._scorers:
            score = scorer.score(term_doc_matrix, model)
            name = scorer.name()
            res[name] = score
        return res
