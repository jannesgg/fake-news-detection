from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import en_coref_lg
import networkx as nx
import pandas as pd
import numpy as np

nlp = en_coref_lg.load()

# paths #

embedding_path = "../../data/embeddings/"

# article summarizer #


def get_embeddings():
    """"""

    # Extract word vectors
    word_embeddings = {}
    f = open(embedding_path + "glove.6B.100d.txt", encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        word_embeddings[word] = coefs
    f.close()

    return word_embeddings


def summarise_fast(article, n=3):
    """"""
    # resolve co-reference issues
    clusters = nlp(article)._.coref_clusters

    if isinstance(clusters, type(None)):
        trans_dict = {}
    else:
        clusters = filter(None, clusters)
        trans_dict = {str(i[1]): str(i[0]) for i in clusters}

    for k, v in trans_dict.items():
        article = article.replace(k, v)

    sentences = sent_tokenize(article)

    return sentences[:n]


def summarise(article, word_embeddings):
    """"""

    sentences = []

    # resolve co-reference issues
    clusters = nlp(article)._.coref_clusters

    if type(clusters) is None:
        trans_dict = {}
    else:
        trans_dict = {str(i[1]): str(i[0]) for i in clusters}

    for k, v in trans_dict.items():
        article = article.replace(k, v)

    sentences.append(sent_tokenize(article))

    sentences = [y for x in sentences for y in x]  # flatten list

    # remove punctuations, numbers and special characters

    if len(sentences) != 0:
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    else:
        return []

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words("english")

    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (
                len(i.split()) + 0.001
            )
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    def compute_PageRank(G, beta=0.85, epsilon=10 ** -4):
        """
        Efficient computation of the PageRank values using a sparse adjacency 
        matrix and the iterative power method.

        Parameters
        ----------
        G : boolean adjacency matrix. np.bool8
            If the element j,i is True, means that there is a link from i to j.
        beta: 1-teleportation probability.
        epsilon: stop condition. Minimum allowed amount of change in the PageRanks
            between iterations.

        Returns
        -------
        output : tuple
            PageRank array normalized top one.
            Number of iterations.

        """
        # Test adjacency matrix is OK
        n, _ = G.shape
        assert G.shape == (n, n)
        # Constants Speed-UP

        # deg_out_beta = G.sum(axis=0).T
        deg_out_beta = (
            G.sum(axis=0).T
            + np.array([[0.0001] for x in np.arange(len(G.sum(axis=0).T))])
        ) / beta  # vector
        deg_out_beta = np.array(deg_out_beta, dtype=np.float64)
        # Initialize
        ranks = np.ones((n, 1)) / n  # vector
        time = 0
        flag = True
        while flag and time < 5:
            time += 1
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # Ignore division by 0 on ranks/deg_out_beta
                ranks = np.array(ranks, dtype=np.float64)
                new_ranks = G.dot((ranks / deg_out_beta))  # vector
            # Leaked PageRank
            new_ranks += (1 - new_ranks.sum()) / n
            # Stop condition
            if np.linalg.norm(ranks - new_ranks, ord=1) <= epsilon:
                flag = False
            ranks = new_ranks
        return {k: v.item(0) for k, v in enumerate(ranks)}

    A = np.matrix(sentence_vectors)
    dist = cosine_similarity(A)
    try:
        nx_graph = nx.from_numpy_array(dist)
    except:
        pass
    mat = nx.adjacency_matrix(nx_graph)

    try:
        scores = compute_PageRank(mat)
    except:
        return []
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )

    return [ranked_sentences[i][1] for i in range(1)]
