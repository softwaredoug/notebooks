import numpy as np
import sys


class WeightsArray:
    """A dynamic array of monotonically increasing row index -> set of weights.

    Intended to replace a dictionary that was taking up a lot of memory that looked like

    weights[(row, col)] = a_float

    Now with this, you're expected to know what a row's columns correspond to, and
    its expected that you'll append a bunch of floats to one row before moving onto the next.

    NOT THREAD SAFE

    """
    def __init__(self):
        self.rows = []  # index of row number into buffer
        self.size = 0    # data array size
        self.data = self._get_buffer()   # array holding col -> row values

    def __len__(self):
        return self.size

    def _get_buffer(self, size=1000):
        return np.zeros(size, dtype=np.float32)

    def _track_rows(self, row):
        assert row >= len(self.rows) - 1
        if row > len(self.rows):
            raise ValueError("Rows should be appended sequentially. You skipped a row!")
        if row > len(self.rows) - 1:
            self.rows.append(self.size)

    def append(self, row: int, value: float):
        """Append with increasing row and col values."""
        self._track_rows(row)
        if self.size == len(self.data):
            new_buffer_size = max(1000, len(self.data) // 3)
            self.data = np.append(self.data, self._get_buffer(size=new_buffer_size))
        self.data[self.size] = value
        self.size += 1

    def append_many(self, row: int, values: np.ndarray):
        self._track_rows(row)
        while (self.size + len(values)) > len(self.data):
            self.data = np.append(self.data, self._get_buffer())
        self.data[self.size:self.size+len(values)] = values
        self.size += len(values)

    def merge(self, other):
        """Append other onto self.

        Assumes you want rows of other to == other.rows+ len(self.rows)
        """
        self_data = self.data[0:self.size]
        other_data = other.data[0:other.size]

        self.data = np.append(self_data, other_data)
        for other_data_idx in other.rows:
            other_data_idx += len(self_data)
            self.rows.append(other_data_idx)
        self.size = len(self.data)

    def weights_of(self, row):
        """Get weights of a row."""
        begin = self.rows[row]
        end = self.size
        if row + 1 < len(self.rows):
            end = self.rows[row+1]
        weights = self.data[begin:end]
        return weights


def test_single_valued_row():
    array = WeightsArray()
    array.append(0, 1.0)

    assert (array.weights_of(0) == np.array([1.0], dtype=np.float32)).all()


def test_row_append():
    array = WeightsArray()
    array.append(0, 1.0)
    array.append(0, 2.0)

    assert (array.weights_of(0) == np.array([1.0, 2.0], dtype=np.float32)).all()


def test_row_append_many():
    array = WeightsArray()
    array.append_many(0, np.array([1.0, 2.0]))

    assert (array.weights_of(0) == np.array([1.0, 2.0], dtype=np.float32)).all()


def test_row_append_many_multiple():
    array = WeightsArray()
    array.append_many(0, np.array([1.0, 2.0]))
    array.append_many(1, np.array([3.0, 4.0, 5.0]))

    assert (array.weights_of(0) == np.array([1.0, 2.0], dtype=np.float32)).all()
    assert (array.weights_of(1) == np.array([3.0, 4.0, 5.0], dtype=np.float32)).all()


def test_row_append_many_multiple_with_singles():
    array = WeightsArray()
    array.append_many(0, np.array([1.0, 2.0]))
    array.append_many(1, np.array([3.0, 4.0, 5.0]))
    array.append(2, 6.0)
    array.append(2, 7.0)

    assert (array.weights_of(0) == np.array([1.0, 2.0], dtype=np.float32)).all()
    assert (array.weights_of(1) == np.array([3.0, 4.0, 5.0], dtype=np.float32)).all()
    assert (array.weights_of(2) == np.array([6.0, 7.0], dtype=np.float32)).all()


def test_row_append_many_multiple_with_singles_then_multiple():
    array = WeightsArray()
    array.append_many(0, np.array([1.0, 2.0]))
    array.append_many(1, np.array([3.0, 4.0, 5.0]))
    array.append(2, 6.0)
    array.append(2, 7.0)
    array.append_many(3, np.array([8.0, 9.0, 10.0]))

    assert (array.weights_of(0) == np.array([1.0, 2.0], dtype=np.float32)).all()
    assert (array.weights_of(1) == np.array([3.0, 4.0, 5.0], dtype=np.float32)).all()
    assert (array.weights_of(2) == np.array([6.0, 7.0], dtype=np.float32)).all()
    assert (array.weights_of(3) == np.array([8.0, 9.0, 10.0], dtype=np.float32)).all()


def test_row_append_multiple():
    array = WeightsArray()
    array.append(0, 1.0)
    array.append(0, 2.0)

    array.append(1, 3.0)
    array.append(1, 4.0)
    array.append(1, 5.0)
    array.append(1, 6.0)

    array.append(2, 7.0)
    array.append(2, 8.0)
    array.append(3, 9.0)
    array.append(4, 10.0)

    assert (array.weights_of(0) == np.array([1.0, 2.0], dtype=np.float32)).all()
    assert (array.weights_of(1) == np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)).all()
    assert (array.weights_of(2) == np.array([7.0, 8.0], dtype=np.float32)).all()
    assert (array.weights_of(3) == np.array([9.0], dtype=np.float32)).all()
    assert (array.weights_of(4) == np.array([10.0], dtype=np.float32)).all()


def test_row_append_out_of_order_fails():
    array = WeightsArray()
    array.append(0, 1.0)
    array.append(0, 2.0)

    array.append(1, 2.0)

    try:
        array.append(3, 4.0)
        raise AssertionError("Exception not thrown for out of order appending.""")
    except ValueError:
        pass


def test_merge_self_rows_with_other():
    array = WeightsArray()
    array.append(0, 1.0)
    array.append(0, 2.0)

    array.append(1, 3.0)
    array.append(1, 4.0)
    array.append(1, 5.0)
    array.append(1, 6.0)

    array.append(2, 7.0)
    array.append(2, 8.0)
    array.append(3, 9.0)
    array.append(4, 10.0)

    other = WeightsArray()
    other.append(0, 11.0)
    other.append(0, 12.0)

    other.append(1, 13.0)

    array.merge(other)

    assert (array.weights_of(0) == np.array([1.0, 2.0], dtype=np.float32)).all()
    assert (array.weights_of(1) == np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)).all()
    assert (array.weights_of(2) == np.array([7.0, 8.0], dtype=np.float32)).all()
    assert (array.weights_of(3) == np.array([9.0], dtype=np.float32)).all()
    assert (array.weights_of(4) == np.array([10.0], dtype=np.float32)).all()
    assert (array.weights_of(5) == np.array([11.0, 12.], dtype=np.float32)).all()
    assert (array.weights_of(6) == np.array([13.0], dtype=np.float32)).all()


def run_my_tests(module: str):
    module = sys.modules[module]
    for member in dir(module):
        if member.startswith('test'):
            function = getattr(module, member)
            if callable(function):
                print(function)
                function()


if __name__ == "__main__":
    run_my_tests(__name__)
