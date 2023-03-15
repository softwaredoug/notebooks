import numpy as np
import math
import pickle
from sys import argv
from time import perf_counter
from pyroaring import BitMap
from project import project
from weights import WeightsArray
# from pympler import asizeof
# All data - quite large for the entire set
# can be downlloaded from
# https://www.kaggle.com/datasets/softwaredoug/wikipedia-sentences-all-minilm-l6-v2


def load_sentences():
    with open('wikisent2_all.npz', 'rb') as f:
        vects = np.load(f)
        vects = vects['arr_0']
        # vects = np.stack(vects)
        all_normed = (np.linalg.norm(vects, axis=1) > 0.99) & (np.linalg.norm(vects, axis=1) < 1.01)
        assert all_normed.all(), "Something is wrong - vectors are not normalized!"

    with open('wikisent2.txt', 'rt') as f:
        sentences = f.readlines()

    return sentences, vects


def centroid(vects):
    """ Sample a unit vector from a sphere in N dimensions.
    It's actually important this is gaussian
    https://stackoverflow.com/questions/59954810/generate-random-points-on-10-dimensional-unit-sphere
    IE Don't do this
        projection = np.random.random_sample(size=num_dims)
        projection /= np.linalg.norm(projection)
    """
    num_dims = len(vects[0])
    projection = np.random.normal(size=num_dims)
    projection /= np.linalg.norm(projection)
    return projection


def most_similar(vects, centroid, floor):
    nn = np.dot(vects, centroid)
    idx_above_thresh = np.sort(np.argwhere(nn >= floor)[:, 0])
    return list(zip(idx_above_thresh, nn[idx_above_thresh]))


class RefsIndex:

    def __init__(self, refs, neighbors, weights):
        self.refs = refs
        self.neighbors = neighbors
        self.weights = weights

    def merge(self, other):
        self.weights.merge(other.weights)

        refs_appended = np.append(self.refs, other.refs, axis=0)
        self.refs = refs_appended

        start_row = len(self.neighbors)
        for ref_idx, neighbors in other.neighbors.items():
            self.neighbors[ref_idx + start_row] = neighbors

    def weights_of(self, ref):
        weights = self.weights.weights_of(ref)
        cols = self.neighbors[ref]
        if len(weights[len(cols):]) > 0:
            assert (weights[len(cols):] == 0.0).all()
        weights = weights[:len(cols)]
        return dict(zip(self.neighbors[ref], weights))

    def search(self, query_vector, ref_points=100, at=10, reweight=True):
        # query vect -> refs similarity
        nn = np.dot(self.refs, query_vector)

        top_n_ref_points = np.argpartition(-nn, ref_points)[:ref_points]
        scored = nn[top_n_ref_points]

        # Top ref candidates via our index
        candidates = {}
        sin_theta = 1.0
        cutoff = 0.0
        refs_so_far = []
        for ref_ord, ref_score in zip(top_n_ref_points, scored):
            sin_theta = 1.0
            if len(refs_so_far) > 0 and reweight:
                refs_span = np.vstack(refs_so_far)
                proj = project(self.refs[ref_ord], refs_span)
                dot = np.dot(proj, self.refs[ref_ord])
                angle = math.acos(dot)
                sin_theta = math.sin(angle)

            for vect_id, score in self.weights_of(ref_ord).items():
                combined = score * ref_score * sin_theta
                if combined > cutoff:
                    try:
                        candidates[vect_id].append(combined)
                    except KeyError:
                        candidates[vect_id] = [combined]

            refs_so_far.append(self.refs[ref_ord])

        summed_candidates = {}
        for vect_id, scored in candidates.items():
            summed_candidates[vect_id] = sum(scored)

        results = summed_candidates.items()
        return sorted(results,
                      key=lambda scored: scored[1],
                      reverse=True)[:at]


def build_index(vects, num_refs=5):
    ref_neighbors = {}   # reference pts -> neighbors
    ref_weights = WeightsArray()    # (ref_ord, vect_ord) -> float

    refs = np.zeros((num_refs, vects.shape[1]))

    all_indexed_vectors = BitMap()

    start = perf_counter()

    for ref_ord in range(0, num_refs):
        specificity = 0.10

        center = centroid(vects)
        top_n = most_similar(vects, center, specificity)

        refs[ref_ord, :] = center
        # idx = []
        bit_set = BitMap()
        for vector_ord, dot_prod in top_n:
            all_indexed_vectors.add(vector_ord)
            bit_set.add(vector_ord)
            ref_weights.append(ref_ord, dot_prod)
        ref_neighbors[ref_ord] = bit_set

        if ref_ord % 10 == 0:
            print(ref_ord, len(ref_neighbors[ref_ord]),
                  len(all_indexed_vectors) / len(vects),
                  perf_counter() - start)

    return RefsIndex(refs, ref_neighbors, ref_weights)


def load_index(filename='index.pkl'):
    with open('index.pkl', 'rb') as f:
        return pickle.load(f)


def test_index_build():
    """Test run of the index build."""
    sentences, vects = load_sentences()
    refs_index1 = build_index(vects, num_refs=10)
    refs_index2 = build_index(vects, num_refs=10)
    refs_index3 = build_index(vects, num_refs=10)

    refs_index1.merge(refs_index2)
    refs_index1.merge(refs_index3)

    copied_15 = refs_index1.weights_of(15)
    orig_5 = refs_index2.weights_of(5)
    assert copied_15 == orig_5

    copied_25 = refs_index1.weights_of(25)
    orig_5 = refs_index3.weights_of(5)
    assert copied_25 == orig_5

    assert (refs_index1.refs[25] == refs_index3.refs[5]).all()


def main(refs=10000):
    sentences, vects = load_sentences()

    refs_per_round = 200
    refs_index = None
    for i in range(0, refs, refs_per_round):
        new_refs_index = build_index(vects, num_refs=refs_per_round)

        if refs_index is None:
            refs_index = new_refs_index
        else:
            refs_index.merge(new_refs_index)

        print(f"{i} - Dumping size {len(refs_index.refs)} refs index")

        with open(f"index_{i}.pkl", 'wb') as f:
            pickle.dump(refs_index, f)


if __name__ == '__main__':
    # test_index_build()
    main(int(argv[1]))
