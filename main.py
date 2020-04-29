import pyximport; pyximport.install()

from topics_model import TopicsModel
from scorers import RelativePerplexityScorer
from regularizers import LDARegularizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import fetch_20newsgroups


if __name__ == '__main__':
    docs = [
        "war politic decision protest corruption",
        "hope end war protest people rights",
        "people kill war arms conflict",
        "destroy bomb people die building",
        "bomb war politic people protest",

        "education student improve plan",
        "university student life study mathematic",
        "lesson student subject exam quality",
        "education quality improve student university",
        "teacher dean people stuff university education",
    ]
    docs_gen = (doc for doc in docs)

    data = fetch_20newsgroups().data
    # data = docs_gen

    analyzer = "word"
    n_topics = 20
    regex1 = '[a-zA-Z]{4,10}'
    # tf_idf_transformer = TfidfTransformer()
    vectorizer = CountVectorizer(analyzer=analyzer, max_features=5000, stop_words="english", token_pattern=regex1)
    term_doc_matrix = vectorizer.fit_transform(data).astype('float64')
    # term_doc_matrix_tf_idf = tf_idf_transformer.fit_transform(term_doc_matrix)
    dictionary = vectorizer.get_feature_names()

    model = TopicsModel(
        n_topics=n_topics,
        dictionary=dictionary,
        scorers=[RelativePerplexityScorer],
        word_topic_reg=[LDARegularizer(alpha=0.01)],
        topic_document_reg=[LDARegularizer(alpha=1.0)],
    )

    for _ in range(100):
        model.fit(term_doc_matrix, batch_size=2000, tol=1e-4, max_iter=30, verbose=2)
        # print(model.score(term_doc_matrix))
        model.print_topics(words_in_topic=10)
