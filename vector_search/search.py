import numpy as np
from index import RefsIndex, load_index, load_sentences  # noqa: F401
from refs_index import warm
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
                           reweight=reweight, ref_points=ref_points,
                           explain_others=[(4433114, vects[4433114])])
    return results


def main(query):
    sentences, vects = load_sentences()

    index = load_index('.old/index_16000_warmed.pkl')
    warm(index, vects, n=10)
    start = perf_counter()
    results = search(index, vects, query,
                     at=10, reweight=True, ref_points=10,
                     as_ref=False)
    for idx, result in enumerate(results[:10]):
        print(idx, result, sentences[result[0]])
    print(f"Results for {query} took {perf_counter() - start}")


if __name__ == "__main__":
    main(argv[1])


"""
X 0 (703438, 0.7017495) Bed Bath & Beyond Inc. is an American chain of domestic merch
andise retail stores in the United States, Puerto Rico, Canada and Mexico.

X 1 (5730694, 0.6105659) The company has been a subsidiary of Bed Bath & Beyond since
 being acquired in 2012.

X 2 (2855157, 0.5598711) It consisted of eight stores when it was acquired by Bed Bat
h & Beyond in 2007.

X 3 (703437, 0.55162036) Bed Bath & Beyond, and Best Buy as junior anchors.

X 4 (5235419, 0.5400942) Steven Howard Temares (born 1958) is an American businessper
son who is the Chief Executive Officer of Bed Bath & Beyond, a national chain of do
mestic merchandise retail stores in both the United States and Canada.

5 (4433114, 0.5224325) Other major tenants include Bed Bath & Beyond, H&M, Launch T
rampoline, Old Navy, Planet Fitness, and Schuler Books & Music.

6 (7741586, 0.4970492) Wherever You Were Last Night'.

7 (3201916, 0.49515733) It is within the archeadconry of Bath.

8 (1333254, 0.49464273) Eastern style bathtubs in which the bather sits up.

9 (7712640, 0.484108) Western style bathtubs in which the bather lies down.
"""
