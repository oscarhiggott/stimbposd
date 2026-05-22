# STIMBPOSD Benchmarks

## sinter collect

Command to use sinter to decode samples from [[72,12,66\]] BB code circuits:

```bash
sinter collect \
    --circuits circuits/bivariatebicyclecodes/r=6,d=6,*,noise=si1000*X,*.stim \
    --decoders bposd bposd-serial bposd-minsum bposd-serial-minsum bplsd bplsd-serial bplsd-minsum bplsd-serial-minsum \
    --custom_decoders_module_function "stimbposd:sinter_decoders" \
    --max_shots 10_000_000 \
    --max_errors 100 \
    --processes auto \
    --metadata_func auto \
    --save_resume_filepath "stats.csv"
```

## Plotting

Plot logical error rate vs physical error rate:

```bash
sinter plot \
    --in stats.csv \
    --x_func "m.p" \
    --failure_unit_name round \
    --failure_units_per_shot_func "m.r" \
    --group_func 'f"{decoder}"' \
    --xaxis "[log] Physical error rate" \
    --yaxis "[log] Logical error rate per round" \
    --title "Logical error rate vs physical error rate for [[72,12,6]] BB code" \
    --highlight_max_likelihood_factor 10 \
    --out "figures/decoder_ler_comparison.png"
```

Plot runtime vs physical error rate:

```bash
sinter plot \
    --in stats.csv \
    --x_func "m.p" \
    --y_func "stat.seconds / stat.shots" \
    --group_func 'f"{decoder}"' \
    --xaxis "[log] Physical error rate" \
    --yaxis "[log] Runtime per shot" \
    --highlight_max_likelihood_factor 10 \
    --title "Decoder runtime vs physical error rate for [[72,12,6]] BB code" \
    --out "figures/decoder_runtime_comparison.png"
```