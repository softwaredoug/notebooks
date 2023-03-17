import numpy as np
import pickle
from sys import argv
import os.path
import glob
from time import perf_counter
from refs_index import RefsIndex
# All data - quite large for the entire set
# can be downlloaded from
# https://www.kaggle.com/datasets/softwaredoug/wikipedia-sentences-all-minilm-l6-v2


def load_sentences():
    # From
    # https://www.kaggle.com/datasets/softwaredoug/wikipedia-sentences-all-minilm-l6-v2
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


def build_index(vects, num_refs=5):
    refs = np.zeros((num_refs, vects.shape[1]))

    index = RefsIndex(dims=vects.shape[1])

    for ref_ord in range(0, num_refs):
        ref = centroid(vects)
        refs[ref_ord, :] = ref

    index.add_refs(refs, vects)

    return index


def load_index(filename='index.pkl'):
    with open(filename, 'rb') as f:
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


def main(refs=10000, last_file=None):
    sentences, vects = load_sentences()

    start_ref = 0
    refs_per_round = 200
    refs_index = None
    if last_file is not None:
        with open(last_file, 'rb') as f:
            refs_index = pickle.load(f)
            start_ref = len(refs_index)

    start = perf_counter()
    for i in range(0, refs, refs_per_round):
        new_refs_index = build_index(vects, num_refs=refs_per_round)

        if refs_index is None:
            refs_index = new_refs_index
        else:
            refs_index.merge(new_refs_index)

        file_num = i + refs_per_round + start_ref
        print(f"{file_num} - Dumping size {len(refs_index.refs)} -- {perf_counter() - start}")

        with open(f"index_{file_num}.pkl", 'wb') as f:
            pickle.dump(refs_index, f)


if __name__ == '__main__':
    last_file = None
    if len(argv) > 2:
        last_file = argv[2]
    main(int(argv[1]), last_file)
