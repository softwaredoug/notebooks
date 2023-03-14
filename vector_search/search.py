from index import RefsIndex, Weights, load_index  # noqa: F401
from sys import argv
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def search(query, at=10, reweight=True):
    index = load_index()
    query_vector = model.encode(query)
    return index.search(query_vector, at=at, reweight=reweight)


def main(query):
    with open('wikisent2.txt', 'rt') as f:
        sentences = f.readlines()

    results = search(query, at=10, reweight=True)
    for idx, result in enumerate(results[:10]):
        print(idx, result, sentences[result[0]])


if __name__ == "__main__":
    main(argv[1])
