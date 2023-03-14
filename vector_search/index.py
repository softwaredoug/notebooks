import numpy as np
import math
import pickle
from time import perf_counter
from pyroaring import BitMap
from project import project
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


class Weights:

    def __init__(self):
        self.rows = []
        self.size = 0
        self.data = self._get_buffer()

    def _get_buffer(self):
        return np.zeros(1000, dtype=np.float32)

    def append(self, row, value):
        """Append with increasing row and col values."""
        assert row >= len(self.rows) - 1
        if row > len(self.rows) - 1:
            self.rows.append(self.size)
        if self.size == len(self.data):
            self.data = np.append(self.data, self._get_buffer())
        self.data[self.size] = value
        self.size += 1

    def weights_of(self, row):
        begin = self.rows[row]
        end = None
        if row + 1 < len(self.rows):
            end = self.rows[row+1]
        weights = self.data[begin:end]
        return weights


class RefsIndex:

    def __init__(self, refs, neighbors, weights):
        self.refs = refs
        self.neighbors = neighbors
        self.weights = weights

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
                print(sin_theta)

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
    ref_weights = Weights()    # (ref_ord, vect_ord) -> float

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


def main():
    sentences, vects = load_sentences()
    refs_index = build_index(vects, num_refs=10000)
    with open('index.pkl', 'wb') as f:
        pickle.dump(refs_index, f)


if __name__ == '__main__':
    main()
