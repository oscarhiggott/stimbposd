from typing import Union
import numpy as np
import stim
from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices

try:
    from ldpc.bplsd_decoder import BpLsdDecoder

    HAS_LSD = True
except ImportError:
    HAS_LSD = False

from stimbposd.config import (
    DEFAULT_MAX_BP_ITERS,
    DEFAULT_BP_METHOD,
)


class BPLSD:
    def __init__(
        self,
        model: stim.DetectorErrorModel,
        max_bp_iters: int = DEFAULT_MAX_BP_ITERS,
        bp_method: str = DEFAULT_BP_METHOD,
        lsd_order: int = 0,
        lsd_method: Union[str, int] = "lsd_0",
        bits_per_step: int = 1,
        always_run_lsd: bool = False,
        **bplsd_kwargs,
    ):
        """Class for decoding stim circuits using belief propagation and localized statistics decoding (BP+LSD).
        This class uses Joschka Roffe's BP+LSD decoder as a subroutine.
        """
        if not HAS_LSD:
            raise ImportError(
                "BpLsdDecoder is not available in your installed ldpc package. "
                "Please upgrade ldpc to version 2.0.0 or later to use BPLSD."
            )

        self._matrices = detector_error_model_to_check_matrices(
            model, allow_undecomposed_hyperedges=True
        )
        self.num_detectors = model.num_detectors
        self.num_errors = model.num_errors

        self._bplsd = BpLsdDecoder(
            pcm=self._matrices.check_matrix,
            max_iter=max_bp_iters,
            bp_method=bp_method,
            error_channel=list(self._matrices.priors),
            lsd_order=lsd_order,
            lsd_method=lsd_method,
            bits_per_step=bits_per_step,
            always_run_lsd=always_run_lsd,
            input_vector_type="syndrome",
            **bplsd_kwargs,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were flipped
        """
        corr = self._bplsd.decode(syndrome)
        return (self._matrices.observables_matrix @ corr) % 2

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data.
        """
        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[
                :, : self.num_detectors
            ]
        predictions = np.zeros(
            (shots.shape[0], self._matrices.observables_matrix.shape[0]), dtype=bool
        )
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions
