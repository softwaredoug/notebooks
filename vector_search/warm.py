import pickle
from sys import argv
import os.path
from refs_index import RefsIndex, warm  # noqa: F401
from index import load_sentences


def main(refs, last_file, specificity):
    sentences, vects = load_sentences()
    directory, file = os.path.split(last_file)
    write_file = file.split('.')[0] + '_warmed.pkl'
    write_file = os.path.join(directory, write_file)

    with open(last_file, 'rb') as f:
        refs_index = pickle.load(f)
    warm(refs_index, vects,
         n=refs, specificity=specificity)

    last_file = last_file.split('.')[0]
    with open(write_file, 'wb') as f:
        pickle.dump(refs_index, f)


if __name__ == '__main__':
    last_file = argv[2]
    specificity = float(argv[3])
    main(int(argv[1]), last_file, specificity)
