import numpy as np
from index import RefsIndex, load_index, load_sentences  # noqa: F401
from sys import argv
from time import perf_counter
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def search(index, vects,
           query, at=10,
           reweight=True, ref_points=10, as_ref=True):
    query_vector = model.encode(query)
    if as_ref:
        new_ref = np.array([query_vector])
        index.add_refs(new_ref, vects)
    results = index.search(query_vector, at=at,
                           reweight=reweight, ref_points=ref_points)
    return results


def main(query):
    sentences, vects = load_sentences()

    index = load_index('.old/index_8000.pkl')
    start = perf_counter()
    results = search(index, vects, query,
                     at=10, reweight=True, ref_points=10,
                     as_ref=False)
    for idx, result in enumerate(results[:10]):
        print(idx, result, sentences[result[0]])
    print(f"Results for {query} took {perf_counter() - start}")


if __name__ == "__main__":
    main(argv[1])
