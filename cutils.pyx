import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport abs
from libc.math cimport log, fabs


@cython.boundscheck(False)
cdef double dot(double[:] array_1, double[:] array_2) nogil:
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef double result = 0
    for i in range(x_max):
        result += array_1[i] * array_2[i]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def likelihood(
    double[:, ::1] term_doc_matrix,
    double[:, ::1] words_topics_proba_matrix,
    double[:, :] topics_documents_proba_matrix):

    cdef long n_documents = term_doc_matrix.shape[0]
    cdef long n_words = term_doc_matrix.shape[1]
    cdef long n_topics = topics_documents_proba_matrix.shape[0]
    cdef double n_wd
    cdef double p_wd

    cdef double result = 0.0
    with nogil:
        for doc_id in range(n_documents):
            for word_id in range(n_words):
                p_wd = dot(words_topics_proba_matrix[word_id], topics_documents_proba_matrix[:, doc_id])
                n_wd = term_doc_matrix[doc_id, word_id]
                result += n_wd * log(p_wd + 1e-16)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def update_triplet_counts(double[:, ::1] phi_mat, double[:, :] tht_mat, double[:, ::1] term_doc_mat,
                          double[:, ::1] n_wt, double[:, ::1] n_td, int doc_start_id, int doc_end_id,
                          int n_topics):
    cdef long doc_id
    cdef size_t word_id, topic_id, dlt_doc_id
    cdef double n_tdw

    with nogil:
        for doc_id in range(doc_start_id, doc_end_id):
            dlt_doc_id = doc_id - doc_start_id
            for word_id in range(term_doc_mat[dlt_doc_id].shape[0]):
                if fabs(term_doc_mat[dlt_doc_id][word_id]) > 1e-16:
                    for topic_id in range(n_topics):
                        n_tdw = term_doc_mat[doc_id - doc_start_id, word_id] * (
                            phi_mat[word_id, topic_id] * tht_mat[topic_id, dlt_doc_id] /
                            (dot(phi_mat[word_id], tht_mat[:, dlt_doc_id]) + 1e-16))
                        n_wt[word_id, topic_id] = n_wt[word_id, topic_id] + n_tdw
                        n_td[topic_id, doc_id] = n_td[topic_id, doc_id] + n_tdw
