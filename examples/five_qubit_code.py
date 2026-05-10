"""[[5,1,3]] perfect code — logical error rate vs physical error rate with BP+OSD.

The five-qubit perfect code encodes 1 logical qubit into 5 physical qubits and can
correct any single-qubit error. This script constructs a syndrome-extraction circuit
for the code, decodes shot data using stimbposd.BPOSD, and plots the logical error
rate as a function of the physical error rate for several numbers of syndrome rounds.

Usage:
    python five_qubit_code.py

Requires: stim, stimbposd, numpy, matplotlib, tqdm
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
import tqdm

from stimbposd import BPOSD

EXAMPLES_DIR = Path(__file__).resolve().parent


def build_five_qubit_circuit(
    rounds: int,
    p_gate: float = 0.0,
    p_meas: float = 0.0,
    p_idle: float = 0.0,
) -> stim.Circuit:
    """Build a [[5,1,3]] syndrome-extraction circuit with depolarising noise.

    Uses 9 qubits: data qubits 0-4 and ancilla qubits 5-8. The four weight-4
    stabiliser generators are:
        g0 = IXZZX   g1 = XIXZZ   g2 = ZXIXZ   g3 = ZZXIX

    Each generator is measured by conjugating the ancilla with H, then applying
    CX (for X-type) and CZ (for Z-type) interactions, then measuring.

    Parameters
    ----------
    rounds : int
        Number of noisy syndrome rounds.
    p_gate : float
        Single- and two-qubit gate depolarising error rate.
    p_meas : float
        Measurement flip probability.
    p_idle : float
        Idle data-qubit depolarising error rate per syndrome round.
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
            # X-type interactions (via CX): qubits i and (i+3) mod 5
            c.append("CX", [anc[i], data[i % 5], anc[i], data[(i + 3) % 5]])
            if pg > 0:
                c.append("DEPOLARIZE2", [anc[i], data[i % 5], anc[i], data[(i + 3) % 5]], pg)
            # Z-type interactions (via CZ): qubits (i+1) and (i+2) mod 5
            c.append("CZ", [anc[i], data[(i + 1) % 5], anc[i], data[(i + 2) % 5]])
            if pg > 0:
                c.append("DEPOLARIZE2", [anc[i], data[(i + 1) % 5], anc[i], data[(i + 2) % 5]], pg)
            c.append("TICK")
        c.append("H", anc)
        if pg > 0:
            c.append("DEPOLARIZE1", anc, pg)
        if pi > 0:
            c.append("DEPOLARIZE1", data, pi)
        c.append("MR", anc, pm)
        return c

    # Noiseless initialisation round to establish reference measurement values
    code += stabilizer_cycle(0.0, 0.0, 0.0)
    code.append("TICK")

    noisy = stabilizer_cycle(p_gate, p_meas, p_idle)
    for i in range(4):
        noisy.append("DETECTOR", [stim.target_rec(-1 - i), stim.target_rec(-5 - i)], [0.5 + i, 2])

    code += rounds * noisy
    code.append("M", data)
    code.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i) for i in range(1, 6)], 0)
    return code


def run_experiment(
    rounds: int,
    shots: int,
    p: float,
    seed: int = 0,
) -> float:
    """Return the logical error rate for one (rounds, p) operating point."""
    code = build_five_qubit_circuit(rounds, p_gate=p, p_meas=p, p_idle=p)
    dem = code.detector_error_model(decompose_errors=True)
    decoder = BPOSD(dem, max_bp_iters=20, osd_order=5)
    sampler = code.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots=shots, separate_observables=True)
    predictions = decoder.decode_batch(det)
    return float(np.mean(np.any(predictions != obs, axis=1)))


def main() -> None:
    p_values = np.linspace(0.001, 0.12, 25)
    rounds_list = [1, 3, 5]
    shots = 20_000

    fig, ax = plt.subplots(figsize=(7, 5))

    for rounds in rounds_list:
        p_logical = [
            run_experiment(rounds, shots, p)
            for p in tqdm.tqdm(p_values, desc=f"rounds={rounds}")
        ]
        ax.semilogy(p_values, p_logical, marker="o", markersize=3, label=f"rounds={rounds}")

    ax.semilogy(p_values, p_values, "k--", label="p_L = p (break-even)")
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate p_L")
    ax.set_title("[[5,1,3]] perfect code — BP+OSD decoder")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out = EXAMPLES_DIR / "five_qubit_code_bposd.pdf"
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved plot to {out}")
    plt.show()


if __name__ == "__main__":
    main()
