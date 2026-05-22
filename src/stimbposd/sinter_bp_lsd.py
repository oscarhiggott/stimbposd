import pathlib
from typing import Union
from sinter import Decoder, CompiledDecoder
import numpy as np
import stim

from stimbposd.bp_lsd import BPLSD, HAS_LSD
from stimbposd.config import (
    DEFAULT_MAX_BP_ITERS,
    DEFAULT_BP_METHOD,
)


class SinterCompiledDecoder_BPLSD(CompiledDecoder):
    def __init__(self, decoder: "BPLSD"):
        self.decoder = decoder

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: "np.ndarray",
    ) -> "np.ndarray":
        return self.decoder.decode_batch(
            shots=bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )


class SinterDecoder_BPLSD(Decoder):
    def __init__(
        self,
        max_bp_iters: int = DEFAULT_MAX_BP_ITERS,
        bp_method: str = DEFAULT_BP_METHOD,
        lsd_order: int = 0,
        lsd_method: Union[str, int] = "lsd_0",
        bits_per_step: int = 1,
        always_run_lsd: bool = False,
        **bplsd_kwargs,
    ):
        """Sinter decoder wrapper for BPLSD."""
        if not HAS_LSD:
            raise ImportError(
                "BpLsdDecoder is not available in your installed ldpc package. "
                "Please upgrade ldpc to version 2.0.0 or later to use BPLSD."
            )
        self.max_bp_iters = max_bp_iters
        self.bp_method = bp_method
        self.lsd_order = lsd_order
        self.lsd_method = lsd_method
        self.bits_per_step = bits_per_step
        self.always_run_lsd = always_run_lsd
        self.bplsd_kwargs = bplsd_kwargs

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledDecoder:
        bplsd = BPLSD(
            model=dem,
            max_bp_iters=self.max_bp_iters,
            bp_method=self.bp_method,
            lsd_order=self.lsd_order,
            lsd_method=self.lsd_method,
            bits_per_step=self.bits_per_step,
            always_run_lsd=self.always_run_lsd,
            **self.bplsd_kwargs,
        )
        return SinterCompiledDecoder_BPLSD(bplsd)

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        dem = stim.DetectorErrorModel.from_file(dem_path)
        bplsd = BPLSD(
            model=dem,
            max_bp_iters=self.max_bp_iters,
            bp_method=self.bp_method,
            lsd_order=self.lsd_order,
            lsd_method=self.lsd_method,
            bits_per_step=self.bits_per_step,
            always_run_lsd=self.always_run_lsd,
            **self.bplsd_kwargs,
        )
        shots = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format="b8",
            num_detectors=dem.num_detectors,
            bit_packed=False,
        )
        predictions = bplsd.decode_batch(shots)
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=dem.num_observables,
        )
