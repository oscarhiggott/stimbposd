from typing import Optional
from pathlib import Path

import pytest
import numpy as np
import stim
from sinter._decoding import sample_decode

from stimbposd import BPOSD, SinterDecoder_BPOSD, sinter_decoders

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize(
    "filename,expected_bposd_errors,expected_vacuous_errors",
    [("bivariate_bicycle_nkd_72_12_6_p_0.001_round_6", 0, 12)],
)
def test_bposd_on_bivariate_bicycle_circuits(
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
        custom_decoders={"bposd": SinterDecoder_BPOSD()},
    )
    assert 1 <= result.errors <= 100
    assert result.shots == 1000


@pytest.mark.parametrize(
    "filename,force_streaming",
    [
        ("bivariate_bicycle_nkd_72_12_6_p_0.001_round_6", True),
        ("bivariate_bicycle_nkd_72_12_6_p_0.001_round_6", False),
    ],
)
def test_sinter_decode_bivariate_bicycle(
    filename: str, force_streaming: Optional[bool]
):
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
        custom_decoders=sinter_decoders(),
    )
    assert result.discards == 0
    assert 0 <= result.errors <= 2
    assert result.shots == 20


def test_noiseless_circuit():
    test_circ = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=25,
        distance=9,
        before_round_data_depolarization=0,
        before_measure_flip_probability=0,
    )
    bposd = BPOSD(test_circ.detector_error_model(), max_bp_iters=20)

    # Test single decode
    syndrome = np.zeros(bposd.num_detectors, dtype=np.uint8)
    prediction = bposd.decode(syndrome)
    assert np.all(prediction == 0)
    assert prediction.shape == (test_circ.num_observables,)

    # Test batch decode
    num_shots = 100
    shots = np.zeros((num_shots, bposd.num_detectors), dtype=np.uint8)
    predictions = bposd.decode_batch(shots)
    assert np.all(predictions == 0)
    assert predictions.shape == (num_shots, test_circ.num_observables)


def test_sinter_decode_noiseless_repetition_code():
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0,
        before_measure_flip_probability=0,
    )
    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        num_shots=1000,
        decoder="bposd",
        custom_decoders={"bposd": SinterDecoder_BPOSD()},
    )
    assert result.errors == 0
    assert result.shots == 1000


def test_sinter_decoders_dict():
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.05,
    )
    decoders = sinter_decoders()
    for decoder_name in decoders:
        result = sample_decode(
            circuit_obj=circuit,
            circuit_path=None,
            dem_obj=circuit.detector_error_model(decompose_errors=True),
            dem_path=None,
            num_shots=100,
            decoder=decoder_name,
            custom_decoders=decoders,
        )
        assert 0 <= result.errors <= 20
        assert result.shots == 100
