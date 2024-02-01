from typing import Tuple
from pathlib import Path

import pytest
import numpy as np
import stim

from stimbposd.bp_osd import BPOSD

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize(
        "filename,expected_bposd_errors,expected_vacuous_errors", 
        [
            ("quasi_cyclic_nkd_72_12_6_p_0.001_round_6", 0, 12)
        ]
        )
def test_bposd_on_quasi_cyclic_circuits(filename: str, expected_bposd_errors: int, expected_vacuous_errors: int):
    circuit = stim.Circuit.from_file(TEST_DATA_DIR / (filename + ".stim"))
    dem = stim.DetectorErrorModel.from_file(TEST_DATA_DIR / (filename + ".dem"))
    bposd = BPOSD(dem, osd_order=5)
    num_dets = circuit.num_detectors
    shot_data = stim.read_shot_data_file(
        path=TEST_DATA_DIR / (filename + ".b8"),
        format="b8",
        num_detectors=num_dets,
        num_observables=circuit.num_observables
    )
    shots, actual_observables = shot_data[:, 0:num_dets], shot_data[:, num_dets:]
    predicted_observables = bposd.decode_batch(shots=shots)
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    assert num_errors == expected_bposd_errors
    num_vacuous_errors = np.sum(np.any(actual_observables, axis=1))
    assert num_vacuous_errors == expected_vacuous_errors
