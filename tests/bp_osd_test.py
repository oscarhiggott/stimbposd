from typing import Optional
from pathlib import Path

import pytest
import numpy as np
import stim
from sinter._decoding import sample_decode

from stimbposd import BPOSD, SinterDecoder_BPOSD, sinter_decoders


def _build_513_circuit(
    rounds: int,
    p_gate: float = 0.0,
    p_meas: float = 0.0,
    p_idle: float = 0.0,
    p_idle_ideal: float = 0.0,
) -> stim.Circuit:
    """Build a [[5,1,3]] syndrome-extraction circuit.

    9 qubits total: 5 data (0-4) and 4 ancilla (5-8).
    Stabilizers: IXZZX, XIXZZ, ZXIXZ, ZZXIX.
    """
    code = stim.Circuit()
    data = list(range(5))
    anc = list(range(5, 9))

    for i in range(5):
        code.append("QUBIT_COORDS", i, [0, i])
    for i in range(4):
        code.append("QUBIT_COORDS", i + 5, [2, 0.5 + i])

    def stabilizer_cycle(pg: float, pm: float, pi: float) -> stim.Circuit:
        c = stim.Circuit()
        c.append("H", anc)
        if pg > 0:
            c.append("DEPOLARIZE1", anc, pg)
        c.append("TICK")
        for i in range(4):
            c.append("CX", [anc[i], data[i % 5], anc[i], data[(i + 3) % 5]])
            if pg > 0:
                c.append("DEPOLARIZE2", [anc[i], data[i % 5], anc[i], data[(i + 3) % 5]], pg)
            c.append("CZ", [anc[i], data[(i + 1) % 5], anc[i], data[(i + 2) % 5]])
            if pg > 0:
                c.append("DEPOLARIZE2", [anc[i], data[(i + 1) % 5], anc[i], data[(i + 2) % 5]], pg)
            c.append("TICK")
        c.append("H", anc)
        if pg > 0:
            c.append("DEPOLARIZE1", anc, pg)
        c.append("MR", anc, pm)
        return c

    # Noiseless initialisation round
    code += stabilizer_cycle(0.0, 0.0, 0.0)
    code.append("TICK")

    noisy = stabilizer_cycle(p_gate, p_meas, p_idle)
    for i in range(4):
        noisy.append("DETECTOR", [stim.target_rec(-1 - i), stim.target_rec(-5 - i)], [0.5 + i, 2])

    if p_idle_ideal > 0:
        code.append("DEPOLARIZE1", data, p_idle_ideal)
    code += rounds * noisy
    code.append("M", data)
    code.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i) for i in range(1, 6)], 0)
    return code

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


def test_bposd_overconstrained_dem():
    """Regression test: BPOSD must not raise when check matrix has more rows than columns.

    The [[5,1,3]] code with idle-only noise produces a DEM with 40 detectors but only
    15 error mechanisms (5 data qubits x 3 Pauli errors), so max_osd_order = 15 - 40 = -25.
    Before the fix, any positive osd_order was clipped to -25 and passed to LDPC, causing:
      ValueError: ERROR: OSD order '-25' invalid. Please choose a positive integer.
    The fix clamps to max(0, min(osd_order, max_osd_order)) so the effective order is 0.
    """
    rounds = 10
    code = _build_513_circuit(rounds, p_idle_ideal=0.01)
    dem = code.detector_error_model(decompose_errors=True)
    # Verify this DEM is indeed overconstrained (more rows than columns)
    from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices
    mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    rows, cols = mats.check_matrix.shape
    assert cols < rows, f"Expected overconstrained DEM (cols={cols} < rows={rows})"
    # Both construction and decoding must succeed without raising
    matcher = BPOSD(dem, max_bp_iters=20, osd_order=5)
    sampler = code.compile_detector_sampler(seed=0)
    det, obs = sampler.sample(shots=100, separate_observables=True)
    predictions = matcher.decode_batch(det)
    assert predictions.shape == obs.shape
