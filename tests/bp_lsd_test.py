import numpy as np
import stim
from sinter._decoding import sample_decode

from stimbposd import BPLSD, SinterDecoder_BPLSD, sinter_decoders


def test_bplsd_decoder():
    test_circ = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=25,
        distance=9,
        after_clifford_depolarization=0.05,
    )
    bplsd = BPLSD(test_circ.detector_error_model(decompose_errors=True))

    # Test single decode
    syndrome = np.zeros(bplsd.num_detectors, dtype=np.uint8)
    prediction = bplsd.decode(syndrome)
    assert np.all(prediction == 0)
    assert prediction.shape == (test_circ.num_observables,)

    # Test batch decode
    num_shots = 100
    sampler = test_circ.compile_detector_sampler()
    shots, observables = sampler.sample(num_shots, separate_observables=True)
    predictions = bplsd.decode_batch(shots)
    assert predictions.shape == (num_shots, test_circ.num_observables)


def test_sinter_decode_bplsd():
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
        num_shots=100,
        decoder="bplsd",
        custom_decoders={"bplsd": SinterDecoder_BPLSD()},
    )
    assert 0 <= result.errors <= 20
    assert result.shots == 100


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
