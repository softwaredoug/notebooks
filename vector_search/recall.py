from index import RefsIndex, Weights, load_index, load_sentences  # noqa: F401
from sys import argv
from search import search
from search_ground_truth import search_ground_truth


def main(queries):
    sentences, vects = load_sentences()
    for query in queries:
        results = search(query, at=10, reweight=True)
        results_gt = search_ground_truth(vects, query, at=10)

        results = set([idx for idx, result in results])
        results_gt = set([idx for idx, result in results_gt])

        print(query, len(results & results_gt))


if __name__ == "__main__":
    main(argv[1:])
