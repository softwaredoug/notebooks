import numpy as np

from index import RefsIndex, load_index, load_sentences  # noqa: F401
from sys import argv
from search import search
from search_ground_truth import search_ground_truth


def warmed(index, vects, n):
    """Sample n random vectors and put them in the index."""
    idxs = np.random.randint(0, len(vects), n)
    new_refs = vects[idxs, :]
    index.add_refs(new_refs, vects, specificity=0.25)


def main(queries):
    print("Loading vects/sentences...")
    sentences, vects = load_sentences()
    print("Loading index...")
    index = load_index('.old/index_8000.pkl')
    # print("Warming...")
    # warmed(index, vects, n=100)
    # print(f"Warmed... {len(vects)}")
    for query in queries:
        results = search(index, vects, query, at=10,
                         ref_points=100, reweight=True,
                         as_ref=False)
        results_gt = search_ground_truth(vects, query, at=10)

        results = set([idx for idx, result, _ in results])
        results_gt = set([idx for idx, result in results_gt])

        print(query, len(results & results_gt))


if __name__ == "__main__":
    main(argv[1:])


"""

(main) $ python search_ground_truth.py "what is the capital of spain
"
0 (3094088, 0.75479484) It is located south of the Spanish capital.

1 (671833, 0.7020078) Barcelona, the capital of Catalonia, is the se
cond largest city and metropolitan area in Spain and sixth-most popu
lous urban area in the European Union.

2 (3923116, 0.69011843) Madrid was named after the capital city of S
pain, Madrid.

3 (671817, 0.68168247) Barcelona is the second largest city of Spain
 and the capital of the autonomous community of Catalonia.

4 (3282967, 0.6729866) Its headquarters is located in Spain.

5 (7321635, 0.6621582) This is a list of currency of Spain.

6 (3564699, 0.6609601) It was written by the singer about the city o
f Madrid, the capital of Spain.

7 (931413, 0.657788) Catalonia and Madrid are the largest communitie
s in Spain in terms of GDP.

8 (3192273, 0.65497464) It is thought to be named for the capital ci
ty of Majorca, an island in the Balearics (Spain), which are located
 south of France.

9 (2975567, 0.65300876) It is also the main Spanish base of the Medi
terranean.
"""
