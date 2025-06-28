import stim
import sinter
import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path

from stimbposd import SinterDecoder_BPOSD, sinter_decoders

EXAMPLES_DIR = Path(__file__).resolve().parent


# Generates surface code circuit tasks
def generate_example_tasks():
    for p in [0.004, 0.006, 0.008, 0.01, 0.012]:
        for d in [3, 5]:
            yield sinter.Task(
                circuit=stim.Circuit.generated(
                    rounds=d,
                    distance=d,
                    after_clifford_depolarization=p,
                    after_reset_flip_probability=p,
                    before_measure_flip_probability=p,
                    before_round_data_depolarization=p,
                    code_task=f"surface_code:rotated_memory_x",
                ),
                json_metadata={
                    "p": p,
                    "d": d,
                },
            )


def main():
    # Collect the samples for stimbposd
    samples = sinter.collect(
        num_workers=multiprocessing.cpu_count() - 1,
        max_shots=500_000,
        max_errors=500,
        tasks=generate_example_tasks(),
        decoders=["bposd"],
        custom_decoders=sinter_decoders(),
        print_progress=True,
    )

    # Plot the data
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f"{stat.decoder}, d={stat.json_metadata['d']}",
        x_func=lambda stat: stat.json_metadata["p"],
    )
    ax.loglog()
    ax.grid()
    ax.set_title("Logical Error Rate vs Physical Error Rate")
    ax.set_ylabel("Logical Error Probability (per shot)")
    ax.set_xlabel("Physical Error Rate")
    ax.legend()
    plt.savefig(EXAMPLES_DIR / "surface_code_bposd.pdf")
    plt.show()


if __name__ == "__main__":
    main()
