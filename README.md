# STIMBPOSD

An implementation of the BP+OSD and BP+LSD decoders for circuit-level noise. This package provides functionality to decode stim circuits using the BP+OSD and BP+LSD decoder implementations from the [LDPC](https://github.com/quantumgizmos/ldpc) python package. 

Included are `stimbposd.BPOSD` and `stimbposd.BPLSD` classes that are configured using a `stim.DetectorErrorModel` and decode shot data, directly outputting predicted observables (without sinter), as well as `stimbposd.SinterDecoder_BPOSD` and `stimbposd.SinterDecoder_BPLSD` classes, which subclass `sinter.Decoder`, for interfacing with sinter.

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

To integrate with [sinter](https://github.com/quantumlib/Stim/tree/main/glue/sample), you can use the `stimbposd.sinter_decoders()` dictionary, which provides decoders compatible with `sinter`.

The package currently supports both BP+OSD and BP+LSD decoders, including serial schedule and min-sum versions:
- `"bposd"`: BP+OSD with default parallel schedule and product-sum updates.
- `"bposd-serial"`: BP+OSD with serial schedule, random seeding, and product-sum updates.
- `"bposd-minsum"`: BP+OSD with parallel schedule and min-sum updates.
- `"bposd-serial-minsum"`: BP+OSD with serial schedule, random seeding, and min-sum updates.
- `"bplsd"`: BP+LSD (Localized Statistics Decoding, see [arXiv:2406.18655](https://arxiv.org/abs/2406.18655)) with default parallel schedule and product-sum updates (requires `ldpc>=2.0.0`).
- `"bplsd-serial"`: BP+LSD with serial schedule, random seeding, and product-sum updates (requires `ldpc>=2.0.0`).
- `"bplsd-minsum"`: BP+LSD with parallel schedule and min-sum updates (requires `ldpc>=2.0.0`).
- `"bplsd-serial-minsum"`: BP+LSD with serial schedule, random seeding, and min-sum updates (requires `ldpc>=2.0.0`).

See the `benchmarks` subdirectory for logical error rate and runtime benchmarks for small bivariate bicycle code circuits. A reasonable choice is "bplsd-serial-minsum", which has a reasonable trade-off of speed and accuracy.


#### Python Usage

You can pass the decoders via the `custom_decoders` argument in `sinter.collect`:

```python
import sinter
from stimbposd import sinter_decoders

# Collect samples using bposd and bposd-serial
samples = sinter.collect(
    num_workers=4,
    max_shots=1_000_000,
    max_errors=1000,
    tasks=generate_example_tasks(),
    decoders=['bposd', 'bposd-serial'],
    custom_decoders=sinter_decoders()
)
```

If you want to use BP+LSD (Localized Statistics Decoding), you can do the same (provided you have `ldpc>=2.0.0` installed):

```python
# Collect samples using bplsd
samples = sinter.collect(
    num_workers=4,
    max_shots=1_000_000,
    max_errors=1000,
    tasks=generate_example_tasks(),
    decoders=['bplsd'],
    custom_decoders=sinter_decoders()
)
```

A complete example using sinter to compare stimbposd with pymatching
can be found in the `examples/surface_code_threshold.py` file (this file also 
includes a definition of `generate_example_tasks()` used above).

#### Command Line Usage

Sinter can also be used from the command line. You can interface `stimbposd` with the `sinter` CLI by using the `--custom_decoders_module_function` argument:

```bash
sinter collect \
    --circuits "example_circuit.stim" \
    --decoders bposd bposd-serial \
    --custom_decoders_module_function "stimbposd:sinter_decoders" \
    --max_shots 100_000 \
    --max_errors 100 \
    --processes auto \
    --save_resume_filepath "stats.csv"
```

Or to use BP+LSD from the command line:

```bash
sinter collect \
    --circuits "example_circuit.stim" \
    --decoders bplsd \
    --custom_decoders_module_function "stimbposd:sinter_decoders" \
    --max_shots 100_000 \
    --max_errors 100 \
    --processes auto \
    --save_resume_filepath "stats.csv"
```

See `benchmarks/benchmark_sinter_commands.md` for an example of how to use the sinter command line to benchmark and plot these decoders on some bivariate bicycle code circuits.

## Performance

BP+OSD has a running time that is cubic in the size of the `stim.DetectorErrorModel` (since the OSD post-processing step involves Gaussian elimination) and is therefore not suitable for very large circuits.

The main advantage of the decoder is that it can be applied to *any* stim circuit and has reasonably good accuracy. It is a heuristic decoder that typically finds low-weight solutions (rather than minimum weight solutions).

See the `benchmarks` subdirectory for some performance data on bivariate bicycle codes.

### Impact of short cycles on decoder performance

The performance of the decoder can be impacted by the presence of many short cycles (e.g. of length less than 6) in the Tanner graph. One common cause of length-four cycles in Tanner graphs of quantum error correcting codes and circuits is Y errors in circuits implementing CSS codes when both $X$ *and* $Z$ checks are annotated as detectors in the circuit. If an $X$ and $Z$ stabiliser commute and overlap, there will be a pair of $Y$ errors on the two qubits in common that anti-commute with both stabilisers (a 4-cycle in the Tanner graph). Depending on the circuit, it can therefore sometimes be beneficial to annotate only $X$ or $Z$ checks when using this package (use whichever basis is needed to predict the annotated logical observables). This also has the benefit of making the DEM significantly smaller, leading to a large speed up of BP+OSD.
