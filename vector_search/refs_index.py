import numpy as np
import math
from collections import Counter
from time import perf_counter
from pyroaring import BitMap
from project import project
from weights import WeightsArray


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


def _most_similar(vects, centroid, floor):
    nn = np.dot(vects, centroid)
    idx_above_thresh = np.sort(np.argwhere(nn >= floor)[:, 0])
    bm = BitMap(idx_above_thresh)
    return bm, nn[idx_above_thresh]


class RefsIndex:

    def __init__(self, dims: int):
        self.refs = None
        self.neighbors = {}
        self.weights = WeightsArray()    # (ref_ord, vect_ord) -> float

    def merge(self, other):
        self.weights.merge(other.weights)

        refs_appended = np.append(self.refs, other.refs, axis=0)
        self.refs = refs_appended

        start_row = len(self.neighbors)
        for ref_idx, neighbors in other.neighbors.items():
            self.neighbors[ref_idx + start_row] = neighbors

    def weights_of(self, ref, multiplier=1.0, n=None):
        """Weights of all ref's neighbors, multiply score by multiplier."""
        weights = np.copy(self.weights.weights_of(ref))
        cols = self.neighbors[ref]
        assert len(weights) == len(cols)
        weights *= multiplier

        if n is not None and len(weights) > n:
            keep = np.argpartition(-weights, n)[:n]
            weights = weights[keep]
            neighbors = self.neighbors[ref]
            neighbors = [neighbors[int(k)] for k in keep]
            return Counter(dict(zip(neighbors, weights)))

        return Counter(dict(zip(self.neighbors[ref], weights)))

    def _append_new_refs(self, refs):
        if self.refs is not None:
            self.refs = np.append(self.refs, refs, axis=0)
        else:
            self.refs = refs

    def __len__(self) -> int:
        if self.refs is None:
            return 0
        return len(self.refs)

    def add_refs(self, refs, vects, specificity=0.1):
        start_ref_ord = len(self)
        self._append_new_refs(refs)
        start = perf_counter()
        for ref_ord, ref in enumerate(refs):
            ref_ord += start_ref_ord
            bit_set, dot_prods = _most_similar(vects, ref, specificity)
            self.weights.append_many(ref_ord, dot_prods)
            self.neighbors[ref_ord] = bit_set
            if ref_ord % 10 == 0:
                print(f"{ref_ord} - {perf_counter() - start}")

    def search(self, query_vector, ref_points=10, at=10, reweight=True):
        # query vect -> refs similarity
        nn = np.dot(self.refs, query_vector)

        ref_points = min(ref_points, len(self.refs))

        if len(nn) == ref_points:
            top_n_ref_points = list(range(0, nn.shape[0]))
            scored = nn
        else:
            top_n_ref_points = np.argpartition(-nn, ref_points)[:ref_points]
            scored = nn[top_n_ref_points]

        # Top ref candidates via our index
        candidates = Counter()
        sin_theta = 1.0
        refs_so_far = []
        for ref_ord, ref_score in zip(top_n_ref_points, scored):
            sin_theta = 1.0
            if len(refs_so_far) > 0 and reweight:
                refs_span = np.vstack(refs_so_far)
                proj = project(self.refs[ref_ord], refs_span)
                dot = np.dot(proj, self.refs[ref_ord])
                angle = math.acos(dot)
                sin_theta = math.sin(angle)

            ref_candidates = self.weights_of(ref_ord, multiplier=ref_score*sin_theta)

            # This line is the bottleneck
            candidates += ref_candidates

            refs_so_far.append(self.refs[ref_ord])

        results = candidates.items()
        return sorted(results,
                      key=lambda scored: scored[1],
                      reverse=True)[:at]


def test_indexing():
    vects = np.array([[1.0, 2.0, 3.0],
                      [-1.0, -2.0, -3.0]])

    vects /= np.linalg.norm(vects, axis=1).reshape(-1, 1)

    index = RefsIndex(dims=3)
    ref = np.array([[0.1, 0.1, 0.1]])
    ref /= np.linalg.norm(ref, axis=1).reshape(-1, 1)
    index.add_refs(ref, vects)

    query_vector = np.array([1.0, 0.9, 0.8])
    query_vector /= np.linalg.norm(query_vector)
    results = index.search(query_vector, ref_points=1)
    assert results[0][0] == 0
    assert results[0][1] >= 0.9 and results[0][1] <= 1.0


def test_indexing_close_refs():
    vects = np.array([[1.0, 2.0, 3.0],
                      [-1.0, -2.0, -3.0]])

    vects /= np.linalg.norm(vects, axis=1).reshape(-1, 1)

    index = RefsIndex(dims=3)
    ref = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.09]])
    ref /= np.linalg.norm(ref, axis=1).reshape(-1, 1)
    index.add_refs(ref, vects)

    query_vector = np.array([1.0, 0.9, 0.8])
    query_vector /= np.linalg.norm(query_vector)
    results = index.search(query_vector, ref_points=1)
    assert results[0][0] == 0
    assert results[0][1] >= 0.9 and results[0][1] <= 1.0


def test_indexing_adjacent_refs():
    vects = np.array([[1.0, 2.0, 3.0],
                      [-1.0, -2.0, -3.0]])

    vects /= np.linalg.norm(vects, axis=1).reshape(-1, 1)

    index = RefsIndex(dims=3)
    ref = np.array([[0.1, 0.1, 0.1], [-1.0, -1.9, -2.9]])
    ref /= np.linalg.norm(ref, axis=1).reshape(-1, 1)
    index.add_refs(ref, vects)

    query_vector = np.array([1.0, 0.9, 0.8])
    query_vector /= np.linalg.norm(query_vector)
    results = index.search(query_vector, ref_points=1)
    assert results[0][0] == 0
    assert results[0][1] >= 0.9 and results[0][1] <= 1.0

    query_vector = np.array([-1.0, -1.9, -2.75])
    query_vector /= np.linalg.norm(query_vector)
    results = index.search(query_vector, ref_points=1)
    assert results[0][0] == 1
    assert results[0][1] >= 0.9 and results[0][1] <= 1.0


def test_add_refs_twice():

    vects = np.array([[1.0, 2.0, 3.0],
                      [-1.0, -2.0, -3.0]])
    vects /= np.linalg.norm(vects, axis=1).reshape(-1, 1)

    index = RefsIndex(dims=3)
    ref = np.array([[-1.0, -1.9, -2.9]])
    ref /= np.linalg.norm(ref, axis=1).reshape(-1, 1)
    index.add_refs(ref, vects)

    vects /= np.linalg.norm(vects, axis=1).reshape(-1, 1)
    ref = np.array([[0.1, 0.1, 0.1]])
    ref /= np.linalg.norm(ref, axis=1).reshape(-1, 1)
    index.add_refs(ref, vects)

    query_vector = np.array([1.0, 0.9, 0.8])
    query_vector /= np.linalg.norm(query_vector)
    results = index.search(query_vector, ref_points=1)
    assert results[0][0] == 0
    assert results[0][1] >= 0.9 and results[0][1] <= 1.0


if __name__ == "__main__":
    test_indexing()
    test_indexing_close_refs()
    test_indexing_adjacent_refs()
    test_add_refs_twice()
