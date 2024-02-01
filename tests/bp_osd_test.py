from typing import Optional
from pathlib import Path

import pytest
import numpy as np
import stim
from sinter._decoding import sample_decode

from stimbposd import BPOSD, BPOSDSinterDecoder

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize(
    "filename,expected_bposd_errors,expected_vacuous_errors",
    [("quasi_cyclic_nkd_72_12_6_p_0.001_round_6", 0, 12)],
)
def test_bposd_on_quasi_cyclic_circuits(
    filename: str, expected_bposd_errors: int, expected_vacuous_errors: int
):
    circuit = stim.Circuit.from_file(TEST_DATA_DIR / (filename + ".stim"))
    dem = stim.DetectorErrorModel.from_file(TEST_DATA_DIR / (filename + ".dem"))
    bposd = BPOSD(dem, osd_order=5)
    num_dets = circuit.num_detectors
    shot_data = stim.read_shot_data_file(
        path=TEST_DATA_DIR / (filename + ".b8"),
        format="b8",
        num_detectors=num_dets,
        num_observables=circuit.num_observables,
    )
    shots, actual_observables = shot_data[:, 0:num_dets], shot_data[:, num_dets:]
    predicted_observables = bposd.decode_batch(shots=shots)
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    assert num_errors == expected_bposd_errors
    num_vacuous_errors = np.sum(np.any(actual_observables, axis=1))
    assert num_vacuous_errors == expected_vacuous_errors


@pytest.mark.parametrize("force_streaming", [True, False])
def test_sinter_decode_repetition_code(force_streaming: Optional[bool]):
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.05,
    )
    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        num_shots=1000,
        decoder="bposd",
        __private__unstable__force_decode_on_disk=force_streaming,
        custom_decoders={"bposd": BPOSDSinterDecoder()},
    )
    assert 1 <= result.errors <= 100
    assert result.shots == 1000


@pytest.mark.parametrize(
    "filename,force_streaming",
    [
        ("quasi_cyclic_nkd_72_12_6_p_0.001_round_6", True),
        ("quasi_cyclic_nkd_72_12_6_p_0.001_round_6", False),
    ],
)
def test_sinter_decode_quasi_cyclic(filename: str, force_streaming: Optional[bool]):
    circuit = stim.Circuit.from_file(TEST_DATA_DIR / (filename + ".stim"))
    dem = stim.DetectorErrorModel.from_file(TEST_DATA_DIR / (filename + ".dem"))
    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=dem,
        dem_path=None,
        num_shots=20,
        decoder="bposd",
        __private__unstable__force_decode_on_disk=force_streaming,
        custom_decoders={"bposd": BPOSDSinterDecoder()},
    )
    assert result.discards == 0
    assert 0 <= result.errors <= 2
    assert result.shots == 20
