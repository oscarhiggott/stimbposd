from typing import List

from scipy.sparse import csc_matrix
import stim
import numpy as np

from stimbposd.dem_to_matrices import (
    iter_set_xor,
    dict_to_csc_matrix,
    detector_error_model_to_check_matrices,
)


def assert_csc_eq(sparse_mat: csc_matrix, dense_mat: List[List[int]]) -> None:
    assert (sparse_mat != csc_matrix(dense_mat)).nnz == 0


def test_iter_set_xor():
    assert iter_set_xor([[0, 1, 2, 5]]) == frozenset((0, 1, 2, 5))
    assert iter_set_xor([[0, 1], [1, 2]]) == frozenset((0, 2))
    assert iter_set_xor([[4, 1, 9, 2], [4, 2, 5, 10]]) == frozenset((1, 5, 9, 10))


def test_dict_to_csc_matrix():
    m = dict_to_csc_matrix(
        {0: frozenset((0, 3)), 1: frozenset((2, 4)), 3: frozenset((1, 3))}, shape=(5, 5)
    )
    assert (
        m
        != csc_matrix(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
            ]
        )
    ).nnz == 0


def test_dem_to_check_matrices():
    dem = stim.DetectorErrorModel(
        """error(0.1) D0 D3 L0 L1 ^ D1 D2 L1
error(0.15) D0 D3 L0 L1 ^ D1 D2 L1
error(0.2) D0 D3 L0 L1
error(0.3) D1 D2 L1
error(0.4) D1 D2 L1"""
    )
    mats = detector_error_model_to_check_matrices(dem)
    assert_csc_eq(mats.check_matrix, [[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]])
    assert_csc_eq(mats.observables_matrix, [[1, 1, 0], [0, 1, 1]])
    assert_csc_eq(mats.edge_check_matrix, [[1, 0], [0, 1], [0, 1], [1, 0]])
    assert_csc_eq(mats.edge_observables_matrix, [[1, 0], [1, 1]])
    assert_csc_eq(mats.hyperedge_to_edge_matrix, [[1, 1, 0], [1, 0, 1]])
    assert np.allclose(mats.priors, np.array([0.22, 0.2, 0.46]))
