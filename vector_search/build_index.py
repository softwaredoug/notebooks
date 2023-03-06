import numpy as np
import pickle
from time import perf_counter
from pyroaring import BitMap
from pympler import asizeof
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
    idx_above_thresh = np.argwhere(nn >= floor)[:, 0].sort()
    import pdb; pdb.set_trace()
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
            self.rows.append(len(self.data))
        if self.size == len(self.data):
            self.data = np.append(self.data, self._get_buffer())
        self.data[self.size] = value
        self.size += 1


class RefsIndex:

    def __init__(self, refs, neighbors, weights):
        self.refs = refs
        self.neighbors = neighbors
        self.weights = weights


def build_index(vects, num_refs=5):
    ref_neighbors = {}   # reference pts -> neighbors
    ref_weights = Weights()    # (ref_ord, vect_ord) -> float

    refs = np.zeros((num_refs, vects.shape[1]))

    all_indexed_vectors = BitMap()

    start = perf_counter()

    for ref_ord in range(0, num_refs):
        specificity = 0.12

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
        if ref_ord % 200 == 0:
            print('W mem ', asizeof.asized(ref_weights, detail=1).format())
            print('N mem ', asizeof.asized(ref_neighbors, detail=1).format())

    return RefsIndex(refs, ref_neighbors, ref_weights)


def main():
    sentences, vects = load_sentences()
    refs_index = build_index(vects, num_refs=8000)
    with open('index.pkl', 'wb') as f:
        pickle.dump(refs_index, f)


if __name__ == '__main__':
    main()
