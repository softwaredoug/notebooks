import numpy as np
from index import load_sentences
from sys import argv
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def search_ground_truth(vects, query, at=10):
    query_vector = model.encode(query)
    nn = np.dot(vects, query_vector)
    top_n = np.argpartition(-nn, at)[:at]
    top_n = top_n[nn[top_n].argsort()[::-1]]
    return sorted(zip(top_n, nn[top_n]),
                  key=lambda scored: scored[1],
                  reverse=True)


def main(query):
    sentences, vects = load_sentences()
    results = search_ground_truth(vects, query, at=10)
    for idx, result in enumerate(results[:10]):
        print(idx, result, sentences[result[0]])


if __name__ == "__main__":
    main(argv[1])
