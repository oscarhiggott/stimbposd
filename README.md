# STIMBPOSD

An implementation of the BP+OSD decoder for circuit-level noise. This package provides functionality to decode stim circuits using the [LDPC](https://github.com/quantumgizmos/ldpc) python package.

Included is a `stimbposd.BPOSD` class that is configured using a `stim.DetectorErrorModel` and decodes shot data, directly outputting predicted observables (without sinter), as well as a `stimbposd.SinterDecoder_BPOSD` class, which subclasses `sinter.Decoder`, for interfacing with sinter.

## Installation

To install from pypi, run:
```
pip install stimbposd
```

To install from source, run:
```
pip install -e .
```
from the root directory.

## Usage

Here is an example of how the decoder can be used directly with Stim:

```python
import stim
import numpy as np
from stimbposd import BPOSD

num_shots = 100
d = 5
p = 0.007
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=d,
    distance=d,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
    after_clifford_depolarization=p
)

sampler = circuit.compile_detector_sampler()
shots, observables = sampler.sample(num_shots, separate_observables=True)

decoder = BPOSD(circuit.detector_error_model(), max_bp_iters=20)

predicted_observables = decoder.decode_batch(shots)
num_mistakes = np.sum(np.any(predicted_observables != observables, axis=1))

print(f"{num_mistakes}/{num_shots}")
```

### Sinter integration

To integrate with [sinter](https://github.com/quantumlib/Stim/tree/main/glue/sample), you can use the 
`stimbposd.SinterDecoder_BPOSD` class, which inherits from `sinter.Decoder`.
To use it, you can use the `custom_decoders` argument when using `sinter.collect`:

```python
import sinter
from stimbposd import SinterDecoder_BPOSD, sinter_decoders

samples = sinter.collect(
    num_workers=4,
    max_shots=1_000_000,
    max_errors=1000,
    tasks=generate_example_tasks(),
    decoders=['bposd'],
    custom_decoders=sinter_decoders()
)
```

A complete example using sinter to compare stimbposd with pymatching
can be found in the `examples/surface_code_threshold.py` file (this file also 
includes a definition of `generate_example_tasks()` used above).



## Performance

BP+OSD has a running time that is cubic in the size of the `stim.DetectorErrorModel` (since the OSD post-processing step involves Gaussian elimination) and is therefore not suitable for very large circuits.

The main advantage of the decoder is that it can be applied to *any* stim circuit and has reasonably good accuracy. It is a heuristic decoder that typically finds low-weight solutions (rather than minimum weight solutions).

### Impact of short cycles on decoder performance

The performance of the decoder can be impacted by the presence of many short cycles (e.g. of length less than 6) in the Tanner graph. One common cause of length-four cycles in Tanner graphs of quantum error correcting codes and circuits is Y errors in circuits implementing CSS codes when both $X$ *and* $Z$ checks are annotated as detectors in the circuit. If an $X$ and $Z$ stabiliser commute and overlap, there will be a pair of $Y$ errors on the two qubits in common that anti-commute with both stabilisers (a 4-cycle in the Tanner graph). Depending on the circuit, it can therefore sometimes be beneficial to annotate only $X$ or $Z$ checks when using this package (use whichever basis is needed to predict the annotated logical observables). This also has the benefit of making the DEM significantly smaller, leading to a large speed up of BP+OSD.
