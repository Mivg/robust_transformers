import uuid
from copy import deepcopy
from unittest import TestCase
import operator as op
import numpy as np

from common.utils import Heap


class HeapTester(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.values = [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        cls.keys = [str(uuid.uuid4()) for _ in range(len(cls.values))]
        cls.inputs = list(zip(cls.keys, cls.values))

    def _run_tests(self, expected_order, key, is_max_heap, keys=None, inputs=None):
        prep_res = op.itemgetter(0) if keys is None else lambda x: x
        keys = keys if keys is not None else self.keys
        inputs = inputs if inputs is not None else self.inputs
        expected_results = [keys[i] for i in expected_order]
        for _ in range(3):
            x = deepcopy(inputs)
            np.random.shuffle(x)
            heap = Heap(x, key, is_max_heap)

            res_one_by_one = [prep_res(heap.pop()) for _ in range(len(expected_order))]
            self.assertEqual(expected_results, res_one_by_one)

            for x_ in x:
                heap.push(x_)
            res_all_after_push = [prep_res(i) for i in heap.nlargest(len(expected_order))]
            self.assertEqual(expected_results, res_all_after_push)

    def test_min_heap_sanity(self):
        # max heap, identity
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 1, 0, 2, 3, 4, 6, 5, 8][::-1]
        key = lambda x: x
        is_max_heap = False
        self._run_tests(expected_order, key, is_max_heap, keys=self.values, inputs=self.values)

    def test_max_heap_sanity(self):
        # max heap, identity
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 1, 0, 2, 3, 4, 6, 5, 8]
        key = lambda x: x
        is_max_heap = True
        self._run_tests(expected_order, key, is_max_heap, keys=self.values, inputs=self.values)

    def test_min_heap_pow_sanity(self):
        # max heap, identity
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 8, 5, 1, 6, 0, 4, 3, 2][::-1]
        key = lambda item: item ** 2
        is_max_heap = False
        self._run_tests(expected_order, key, is_max_heap, keys=self.values, inputs=self.values)

    def test_max_heap_pow_sanity(self):
        # max heap, identity
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 8, 5, 1, 6, 0, 4, 3, 2]
        key = lambda item: item ** 2
        is_max_heap = True
        self._run_tests(expected_order, key, is_max_heap, keys=self.values, inputs=self.values)

    def test_min_heap_identity(self):
        # max heap, identity key (the value item)
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 1, 0, 2, 3, 4, 6, 5, 8][::-1]
        key = op.itemgetter(1)
        is_max_heap = False
        self._run_tests(expected_order, key, is_max_heap)

    def test_max_heap_identity(self):
        # max heap, identity key (the value item)
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 1, 0, 2, 3, 4, 6, 5, 8]
        key = op.itemgetter(1)
        is_max_heap = True
        self._run_tests(expected_order, key, is_max_heap)

    def test_min_heap_pow(self):
        # max heap, identity key (the value item)
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 8, 5, 1, 6, 0, 4, 3, 2][::-1]
        key = lambda item: item[1] ** 2
        is_max_heap = False
        self._run_tests(expected_order, key, is_max_heap)

    def test_max_heap_pow(self):
        # max heap, identity key (the value item)
        # [1.5, 2.7, 0., -.01, -0.2, -3, -2, 5, -4]
        # [0    1    2    3     4     5   6  7   8
        expected_order = [7, 8, 5, 1, 6, 0, 4, 3, 2]
        key = lambda item: item[1] ** 2
        is_max_heap = True
        self._run_tests(expected_order, key, is_max_heap)