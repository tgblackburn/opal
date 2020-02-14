# opal

Parallel, relativistic 1d3v PIC code. Written primarily to do some plasma physics in the strong-field QED regime, but also to test Rust as a platform for HPC.

## Build

The following need to be installed:

* [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/)
* [GSL](https://www.gnu.org/software/gsl/)
* an MPI library

Opal has been tested with OpenMPI.

All other dependencies are Rust crates that are downloaded automatically by Cargo. Then building should be as simple as:

```bash
cargo build --release [-j NUM_THREADS] [--features FEATURES]
```

where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn. `FEATURES` is a comma-separated list of compile-time options (listed in [Cargo.toml](Cargo.toml)) that are disabled by default.

## Specify problem

opal takes as its single argument the path to a YAML file describing the input configuration. Output is automatically written to the same directory as this file. The inputs for some test problems can be found in [examples](examples). Starting from scratch, the input needs to contain the following sections:

* control
* qed
* electrons
* ions

and optionally

* photons (if photon emission or absorption are enabled)
* constants

The structure of the input file is described in detail [here](docs/input.md).

## Run

opal has a hybrid parallelization scheme, using MPI to split the grid into subdomains, and Rayon for each subdomain.

Assuming Opal has been downloaded to `opal_directory` and already built,

```bash
cd opal_directory
export RAYON_NUM_THREADS=nt
mpirun -n np ./target/release/opal path/to/input.yaml
```

will run Opal, distributing the domain over `np` MPI tasks, and assigning `nt` theads per task. The right balance between the number of tasks and threads is system-dependent.

