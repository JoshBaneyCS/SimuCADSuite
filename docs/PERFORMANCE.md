# Performance & Benchmarks

SimuCADSuite includes a Criterion-based benchmark suite that covers the major
computational subsystems. Benchmarks live alongside the crates they exercise and
are collected by `cargo bench --workspace`.

## Running benchmarks locally

```bash
# Run the full suite
cargo bench --workspace
```

Results are written to `target/criterion/` with HTML reports you can open in a
browser.

## Comparing against a baseline

Save a baseline before making changes, then compare after:

```bash
# Save the current state as "base"
cargo bench -- --save-baseline base

# ... make your changes ...

# Run again, comparing to the saved baseline
cargo bench -- --baseline base
```

Criterion will report the percentage change for each benchmark and flag
statistically significant regressions.

## Benchmark groups

| Group | What it measures |
|-------|-----------------|
| **Physics** | Particle advection, spatial grid build and query |
| **CAS** | Expression parse, evaluate, differentiate, simplify |
| **Audio** | FFT, STFT, power spectrum computation |
| **Mesh** | BVH build and query, mesh quality metrics |
| **GPU** | CPU backend advection, velocity field evaluation |

## Detailed comparison with critcmp

For side-by-side comparison of two saved baselines, install
[critcmp](https://github.com/BurntSushi/critcmp):

```bash
cargo install critcmp

# Save two baselines
cargo bench -- --save-baseline before
# ... make changes ...
cargo bench -- --save-baseline after

# Compare
critcmp before after
```

`critcmp` produces a compact table that makes it easy to spot regressions across
the entire suite at a glance.
