# STIMBPOSD

An implementation of the BP+OSD decoder for circuit-level noise. This package provides functionality to decode stim circuits using the [LDPC](https://github.com/quantumgizmos/ldpc) python package.

Included is a `stimbposd.BPOSD` class that is configured using a `stim.DetectorErrorModel` and decodes shot data, directly outputting predicted observables (without sinter), as well as a `stimbposd.BPOSDSinterDecoder` class, which subclasses `sinter.Decoder`, for interfacing with sinter.

## Performance

BP+OSD has a running time that is cubic in the size of the `stim.DetectorErrorModel` (since the OSD post-processing step involves Gaussian elimination) and is therefore not suitable for very large circuits.

The main advantage of the decoder is that it can be applied to *any* stim circuit and has reasonably good accuracy. It is a heuristic decoder that typically finds low-weight solutions (rather than minimum weight solutions).

### Impact of short cycles on decoder performance

The performance of the decoder is heavily impacted by the presence of many short cycles (e.g. of length less than 6) in the Tanner graph. One very common cause of length-four cycles in Tanner graphs of quantum error correcting codes and circuits is Y errors in circuits implementing CSS codes when both $X$ *and* $Z$ checks are annotated as detectors in the circuit. *It is therefore strongly recommended that only $X$ or $Z$ checks are annotated in circuits when using this package (use whichever basis is needed to predict the annotated logical observables).* This also has the benefit of making the DEM significantly smaller, leading to a large speed up of BP+OSD.

More specifically, if an $X$ check and a $Z$ check have any qubits in common in their support, they must overlap on at least two qubits ($q_i$ and $q_j$, say) to commute. Let $D_X$ and $D_Z$ be detectors measuring the $X$ and $Z$ check, respectively, and suppose that the errors $Y_i$ and $Y_j$ exist in the error model. The Tanner graph of the code will therefore have the 4-cycle ($D_X$, $Y_i$, $D_Z$, $Y_j$), since $Y_i$ and $Y_j$ each trigger both detectors. These 4-cycles are often ubiquitous and significantly degrade BP, and so removing either the $X$ or $Z$ checks (whichever is not needed to predict the observables) can improve accuracy, even though this technically removes information from the DEM.
