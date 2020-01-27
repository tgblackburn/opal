# opal
Parallel, relativistic 1d3v PIC code. Written primarily to do some plasma physics in the strong-field QED regime, but also to test Rust as a platform for HPC.

# Build

The following need to be installed:
* [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/)
* [GSL](https://www.gnu.org/software/gsl/)
* an MPI library

Opal has been tested with OpenMPI.

All other dependencies are Rust crates that are downloaded automatically by Cargo. Then building should be as simple as:

```bash
cargo build --release [-j NUM_THREADS]
```
where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn.

# Specify problem
opal takes as its single argument the path to a YAML file describing the input configuration. Output is automatically written to the same directory as this file. The inputs for some test problems can be found in `examples/`. Starting from scratch, the input needs to contain the following sections:

* control
* qed
* electrons
* ions

and optionally

* photons (if photon emission or absorption are enabled)
* constants

## control
* `dx`: grid step (in metres).
* `nx`: number of grid cells.
* `xmin`: x position of the left-hand boundary.
* `start`: time at which simulation starts  (in seconds)...
* `end`: and ends.
* `current_deposition`: if `true`, charged particles create electromagnetic fields according to Maxwell's equations; if `false`, they will only move in response to externally injected fields.
* `output_frequency`: produce this many outputs, at regular intervals.
* `balance` (optional, default is `true`): partition the domain in such a way that all MPI tasks get approximately the same number of particles. This is done once only, at the start of the simulation.

## qed
This determines the extra physics that is included in the simulation.
* `photon_emission`: if `true`, photons are emitted by electrons according to the quantum synchrotron rates.
* `photon_absorption`: if `true`, photons can be absorbed by electrons. This requires pairwise checking of the absorption cross section and signficantly increases simulation runtime.
* `photon_energy_min` (optional, default is `0.0`): specifies an energy (in joules) below which photons are deleted from the simulation on creation. The conversion constants `eV`, `keV` and `MeV` are provided for convenience.

## electrons
At present, opal supports a single electron species. Only `npc` and `output` are actually compulsory, but if `npc` is greater than zero, all other values must be given.
* `npc`: create this number of macroelectrons per grid cell. Larger numbers means better statistics but proportionally longer runtimes.
* `ne`: the initial number density as a function of `x`; the code recognizes most mathematical functions and will look for unknown values in the [constants](#constants) block. The maths parser is explained [here](#maths-parser).
* `ux`: specifies the x-component of the initial momentum in units of the electron mass; as for `ne`, this should be a function of `x`. A temperature (i.e. a randomly distributed `ux`) is possible by use of the special values `urand` and `nrand`, which generate uniformly and normally distributed numbers, respectively.
* `uy`
* `uz`
* `output`: list of specifiers, each of which should correspond to a distribution function. For example, `x:px` requests the distribution of the x coordinate and the corresponding momentum component. Each separate output is written to its own FITS file.

## ions
Similarly, opal supports only one ion species at present. Only `npc` is compulsory, but if it is greater than zero, all other values must be given.
* `npc`: as for [electrons](#electrons).
* `name`: how this ion species should be identified in output.
* `Z`: atomic number, also ionization level.
* `A`: mass number.
* `ni`: as for [electrons](#electrons).
* `ux`: as for [electrons](#electrons), but in units of the ion mass.
* `uy`
* `uz`
* `output`: as for [electrons](#electrons).

## photons
If either `photon_emission` or `photon_absorption` are enabled, this section must exist. However, if `npc` is zero, only `output` needs to be given. Otherwise, the fields are the same as for the [electrons](#electrons), but `ne` is renamed `nph`.

## laser
* `Ey`
* `Ez`

Both must be given as functions of `t` and `x`; they control the value of the electric field at the left-hand boundary of the simulation domain.

## constants
Everywhere an integer or floating-point number is requested in the input file, a named value may be given instead, provided that its value is specified in this section.

For example, `ne: n0 * step(x, 0.0, 1.0e-6)` in the [electrons](#electrons) section would be accepted provided that `n0: 1.0e23` was given. Named constants can themselves be mathematical expressions, but they cannot depend on each other, or themselves.

## Maths parser
The code makes use of [meval](https://crates.io/crates/meval) when parsing the input file. In addition to the functions and constants this crate provides, opal provides:
* `critical(omega)`: returns the critical density (in units of 1/m^3) for the corresponding angular frequency (given in units of rad/s).
* `gauss(x,mu,sigma)`: probability density function of the normal distribution, for mean `mu` and standard deviation `sigma`.
* `step(x,min,max)`: Heaviside theta function, returns 1.0 for min <= x < max and zero otherwise.
* `nrand`: returns a pseudorandom number, normally distributed with mean 0.0 and standard deviation 1.0.
* `urand`: returns a pseudorandom number, uniformly distributed between 0.0 and 1.0.
* the physical constants `me`, `mp`, `c`, `e`: the electron mass, proton mass, speed of light and elementary charge, respectively, all in SI units.
* the conversion constants `eV`, `keV`, `MeV`, `femto`, `pico`, `nano`, `milli`.

# Run
opal has a hybrid parallelization scheme, using MPI to split the grid into subdomains, and Rayon for each subdomain.

Assuming Opal has been downloaded to `opal_directory` and already built,

```bash
cd opal_directory
export RAYON_NUM_THREADS=nt
mpirun -n np ./target/release/opal path/to/input.yaml
```
will run Opal, distributing the domain over `np` MPI tasks, and assigning `nt` theads per task. The right balance between the number of tasks and threads is system-dependent.
