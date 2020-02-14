//! Particle dynamics and interactions

use std::fmt::{Debug};
use mpi::traits::*;
use rand::prelude::*;
use rand_distr::{StandardNormal, Exp1};
use rayon::prelude::*;

mod electron;
mod photon;
mod ion;
mod vec3;
mod hgram;
mod interactions;

// Re-export for use in main
pub use self::electron::*;
pub use self::photon::*;
pub use self::ion::*;
pub use self::hgram::*;
pub use self::interactions::*;

// For local use
use crate::grid::Grid;

/// A Particle represents a specific member of a species,
/// whose dynamics can be modelled by a particle-in-cell simulation.
#[allow(non_snake_case)]
pub trait Particle: Copy + Clone + Debug + Equivalence + PartialOrd {
    /// Creates a macroparticle
    /// - located in the specified `cell`,
    /// - with fractional offset `x` from the cell left-hand boundary,
    /// - with normalized momentum (p/mc) `u`,
    /// - specified `weight` (number of real particles represented),
    /// - on a Grid that has spacing `dx`,
    /// - which is advanced with timestep `dt`.
    fn create(cell: isize, x: f64, u: &[f64; 3], weight: f64, dx: f64, dt: f64) -> Self;

    /// Returns a new particle which has optical depth (against
    /// the default QED process) set to `tau`.
    /// Generally called with a pseudorandom argument.
    fn with_optical_depth(&self, tau: f64) -> Self;

    /// Returns a triple of
    /// - the particles current `cell`,
    /// - its *current* fractional offset from the cell left-hand boundary
    /// - the fractional offset from the same boundary at the previous timestep.
    fn location(&self) -> (isize, f64, f64);

    /// The total displacement in the perpendicular direction, since
    /// the particle was created.
    fn transverse_displacement(&self) -> f64;

    /// Advances the particle momentum and position, using the specified
    /// electric and magnetic fields `E` and `B`, over an interval `dt`.
    /// The grid spacing `dx` needs to be given so that the offset can
    /// be appropriately normalized.
    fn push(&mut self, E: &[f64; 3], B: &[f64; 3], dx: f64, dt: f64);

    /// Particles that cross a subdomain boundary need their cell
    /// index reset to account for this fact. Equivalent to:
    /// ```
    /// pt.cell = pt.cell + delta;
    /// ```
    fn shift_cell(&mut self, delta: isize);

    /// Returns the charge of relevant particle species (not the
    /// total charge of the macroparticle).
    fn charge(&self) -> f64;

    /// Returns the mass of the relevant particle species (not the
    /// total mass of the macroparticle).
    fn mass(&self) -> f64;

    /// Returns the three-velocity of the particle.
    fn velocity(&self) -> [f64; 3];

    /// Returns the relativistic energy (in MeV), equivalent to the particle velocity.
    fn energy(&self) -> f64;

    /// Returns the relativistic momentum (in MeV/c), equivalent to the particle velocity.
    fn momentum(&self) -> [f64; 3];

    /// Returns the four-momentum of this particle, normalized to its mass
    /// and the speed of light.
    fn normalized_four_momentum(&self) -> [f64; 4];

    /// Returns the work done by the electromagnetic field over the
    /// particle trajectory (in joules), for a single particle
    /// with the velocity. Scale by weight() to obtain the work done
    /// on the macroparticle.
    fn work(&self) -> f64;

    /// Returns the kinetic energy of the macroparticle, in joules.
    fn total_kinetic_energy(&self) -> f64;

    /// Returns the weight of the macroparticle, i.e. the number of real particles
    /// it represents.
    fn weight(&self) -> f64;

    /// Returns the quantum nonlinearity parameter for this particle.
    fn chi(&self) -> f64;

    /// Flags this particle, for user-defined purposes.
    fn flag(&mut self);

    /// Removes the flag from this particle, if already flagged.
    fn unflag(&mut self);

    /// Tests if this particle has been flagged for some purpose.
    fn is_flagged(&self) -> bool;

    /// If this particle has a spin property, return its value.
    fn spin_state(&self) -> Option<f64> {
        None
    }

    /// Identify the spin property for a particle of this species,
    /// if one exists.
    fn spin_state_name(&self) -> Option<&'static str> {
        None
    }

}

/// A Population<T> is the principal means by which the main
/// simulation loop interacts with particles of type T.
/// By default, Populations know how to identify, advance
/// and print information about themselves.
/// Particle-field and particle-particle interactions are
/// modelled by coupling concrete implementations of
/// the Population trait.
pub struct Population<T: Particle> {
    store: Vec<T>,
    output: Vec<String>,
    name: String,
}

impl<T> Population<T> where T: Particle + Send + Sync {
    /// Return an immutable slice of all the particles owned
    /// by `self`.
    pub fn all(&self) -> &[T] {
        &self.store[..]
    }

    /// Creates an empty, anonymous population of the
    /// given species.
    pub fn new_empty() -> Population<T> {
        let v: Vec<T> = Vec::new();
        Population {
            store: v,
            output: Vec::new(),
            name: String::with_capacity(16),
        }
    }

    /// Creates a population of particles T, with
    /// - `npc` macroparticles per cell,
    /// - number density as a function of `x`, `number_density`,
    /// - normalized momentum components `ux`, `uy` and `uz`, all possibly functions of `x`,
    /// - on the specified `grid`,
    /// - using the random number generator `rng`,
    /// - where the simulation timestep is `dt`.
    pub fn new<F1,F3,G,R>(npc: usize, number_density: F1, ux: F3, uy: F3, uz: F3, grid: &G, rng: &mut R, dt: f64) -> Population<T>
    where F1: Fn(f64) -> f64, F3: Fn(f64, f64, f64) -> f64, G: Grid, R: Rng
    {
        let mut pt: Vec<T> = Vec::new();
        //let mut rng = rand::thread_rng();

        for c in 0..(grid.size() as isize) {
            let x = grid.xmin() + ((c as f64) + 0.5) * grid.dx();
            let nreal = number_density(x) * grid.dx(); // number of real particles
            let weight = nreal / (npc as f64);
            let nsuper = if nreal > 0.0 {npc} else {0};
            let mut sub: Vec<T> = (0..nsuper)
                .map(|_i| {
                    //let u = normalized_momentum(x);
                    let x = rng.gen();
                    let real_x = grid.xmin() + ((c as f64) + x) * grid.dx();
                    let u = [
                        ux(real_x, rng.gen(), rng.sample(StandardNormal)),
                        uy(real_x, rng.gen(), rng.sample(StandardNormal)),
                        uz(real_x, rng.gen(), rng.sample(StandardNormal)),
                    ];
                    T::create(c, x, &u, weight, grid.dx(), dt).with_optical_depth(rng.sample(Exp1))
                })
                .collect();
            pt.append(&mut sub);
        }
        Population {
            store: pt,
            output: Vec::new(),
            name: String::with_capacity(16)
        }
    }

    /// Applies the function `f` to all particles in this population.
    pub fn map_in_place<F: Fn(&mut T)>(&mut self, f: F) -> &mut Self {
        self.store.iter_mut().for_each(f);
        self
    }

    /// Specifies that this population should write output as given by `ospec`.
    pub fn with_output(&mut self, ospec: Vec<String>) -> &mut Self {
        self.output = ospec;
        self
    }

    /// Specifies the `name` of the population. Singular by default.
    pub fn with_name(&mut self, name: &str) -> &mut Self {
        self.name = name.to_owned();
        self
    }

    /// Returns the total kinetic energy, of all Populations of this type,
    /// across all grids, in joules.
    /// - Must be called on all processes.
    #[allow(non_snake_case)]
    pub fn total_kinetic_energy(&self, comm: &impl Communicator) -> f64 {
        use mpi::collective::SystemOperation;

        let local = self.store.par_iter().map(|pt| pt.total_kinetic_energy()).sum::<f64>();

        if comm.rank() == 0 {
            let mut global = 0.0;
            comm.process_at_rank(0).reduce_into_root(&local, &mut global, SystemOperation::sum());
            global
        } else {
            comm.process_at_rank(0).reduce_into(&local, SystemOperation::sum());
            local
        }
    }

    /// Advance this Population by time `dt`, using the electromagnetic field
    /// stored on the local subdomain `grid`.
    /// Then exchange particles that have crossed a subdomain boundary.
    /// - Must be called on all processes.
    #[allow(non_snake_case)]
    pub fn advance<C,G>(&mut self, comm: &C, grid: &G, dt: f64)
    where C: Communicator, G: Grid + Sync {
        let dx = grid.dx();
        let max = grid.size() as isize;
        let num = self.store.len();
        let tag: i32 = self.name.as_bytes().iter().map(|&b| b as i32).sum::<i32>();

        let nthreads = rayon::current_num_threads();
        // chunk length cannot be zero
        let chunk_len = if num > nthreads {
            num / nthreads
        } else {
            1 // guarantee this runs
        };

        let mut counts: Vec<(usize, usize)> = Vec::with_capacity(nthreads);
        self.store
            .par_chunks_mut(chunk_len)
            .map(|chunk: &mut [T]| -> (usize, usize) {
                let mut gone_left: usize = 0;
                let mut gone_right: usize = 0;
                chunk.iter_mut().for_each(|pt| {
                    let (c, x, _) = pt.location();
                    let (E, B) = grid.fields_at(c, x);
                    pt.push(&E, &B, dx, dt);
                    pt.unflag();
                    let (c, _, _) = pt.location();
                    if c < 0 {
                        gone_left = gone_left + 1;
                    } else if c >= max {
                        gone_right = gone_right + 1;
                    }
                });
                (gone_left, gone_right)
            })
            .collect_into_vec(&mut counts);

        let (gone_left, gone_right) = counts.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

        // sort by GridCell id
        // pretty expensive...
        self.store.par_sort_unstable_by_key(|pt| pt.location().0);

        // Particles that are going to the right need their cell index adjusting
        for pt in &mut self.store[num-gone_right..num] {
            pt.shift_cell(-max);
        }

        // Get slices of lost particles - but don't replace just yet
        let send_left = &self.store[0..gone_left];
        let send_right = &self.store[num-gone_right..num];
        
        // Delete lost particles
        //let send_right: Vec<T> = self.store.drain(num-gone_right..num).collect();
        //let send_left: Vec<T> = self.store.drain(0..gone_left).collect();

        // Even grids go first, sending to right

        let mut recv_left: Vec<T> = Vec::new();
        let mut recv_right: Vec<T> = Vec::new();

        if grid.rank() % 2 == 0 {
            if let Some(r) = grid.to_right() {
                comm.process_at_rank(r).synchronous_send_with_tag(send_right, tag);
                let mut tmp = comm.process_at_rank(r).receive_vec_with_tag::<T>(tag).0;
                recv_right.append(&mut tmp);
            }
        } else {
            if let Some(l) = grid.to_left() {
                let mut tmp = comm.process_at_rank(l).receive_vec_with_tag::<T>(tag).0;
                recv_left.append(&mut tmp);
                comm.process_at_rank(l).synchronous_send_with_tag(send_left, tag);
            }
        }

        // and then they send left

        if grid.rank() % 2 == 0 {
            if let Some(l) = grid.to_left() {
                comm.process_at_rank(l).synchronous_send_with_tag(send_left, tag);
                let mut tmp = comm.process_at_rank(l).receive_vec_with_tag::<T>(tag).0;
                recv_left.append(&mut tmp);
            }
        } else {
            if let Some(r) = grid.to_right() {
                let mut tmp = comm.process_at_rank(r).receive_vec_with_tag::<T>(tag).0;
                recv_right.append(&mut tmp);
                comm.process_at_rank(r).synchronous_send_with_tag(send_right, tag);
            }
        }

        /*
        if recv_left.len() != 0 {
            println!("{} got {} pts from the left", grid.rank(), recv_left.len());
        }
        if recv_right.len() != 0 {
            println!("{} got {} pts from the right", grid.rank(), recv_right.len());
        }
        */

        // Need to correct cells only for particles received from the right!
        for pt in recv_right.iter_mut() {
            pt.shift_cell(max);
        }

        // Overwrite lost particles with new ones
        // Append first!
        self.store.splice(num-gone_right..num, recv_right.iter().cloned());
        self.store.splice(0..gone_left, recv_left.iter().cloned()); // eat the cost, which is not so bad

        //assert!(num - gone_left - gone_right + recv_left.len() + recv_right.len() == self.store.len());
    }

    /// Constructs the requested distribution functions and writes them to
    /// the specified `directory`. Each set of output is identified by the
    /// given `index` and the Population name.
    /// - Must be called on all processes.
    /// - At the moment, distribution functions can be, at most, two-dimensional.
    pub fn write_data<C,G>(&self, comm: &C, grid: &G, directory: &str, index: usize) -> std::io::Result<()> 
    where C: Communicator, G: Grid {
        use std::io::{Error, ErrorKind};
        // self.output = ["f", "f:g", "f:g:(h)", "f:g:(h;a)"]

        let position = |pt: &T| -> f64 {
            let (c, x, _) = pt.location();
            grid.xmin() + ((c as f64) + x) * grid.dx()
        };

        let radius = |pt: &T| -> f64 {
            pt.transverse_displacement()
        };

        let energy = |pt: &T| -> f64 {
            pt.energy()
        };

        let work = |pt: &T| -> f64 {
            pt.work()
        };

        let px = |pt: &T| -> f64 {
            pt.momentum()[0]
        };

        let py = |pt: &T| -> f64 {
            pt.momentum()[1]
        };

        let pz = |pt: &T| -> f64 {
            pt.momentum()[2]
        };

        let p_perp = |pt: &T| -> f64 {
            let p = pt.momentum();
            p[1].hypot(p[2])
        };

        let chi = |pt: &T| -> f64 {
            pt.chi()
        };

        // Polar angle around x-axis
        let polar_angle = |pt: &T| -> f64 {
            let p = pt.momentum();
            let magnitude = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
            f64::acos(p[0] / magnitude)
        };

        // Right-handed azimuthal angle around x-axis, zero directed along y
        let azimuthal_angle = |pt: &T| -> f64 {
            let p = pt.momentum();
            f64::atan2(p[2], p[1])
        };

        // Longitude and latitude (0,0) directed along negative x-axis
        let longitude = |pt : &T| -> f64 {
            let p = pt.momentum();
            f64::atan2(p[1], -p[0]) // atan2[(y, x) = (py, -px)]
        };

        let latitude = |pt: &T| -> f64 {
            let p = pt.momentum();
            let magnitude = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
            f64::asin(p[2] / magnitude) // asin(pz/p)
        };

        for o in &self.output {
            // break into substrings, separated by colons
            let mut ss: Vec<&str> = o.split(':').collect();
            // if the final string is bracketed AND there are at least
            // two substrings, that final string might be (bspec; hspec)
            let (bspec, hspec, weight) = if ss.len() >= 2 && ss.last().unwrap().starts_with('(') && ss.last().unwrap().ends_with(')') {
                let last = ss.pop().unwrap().trim_start_matches('(').trim_end_matches(')');
                // break this into substrings, separated by ';'
                let last: Vec<&str> = last.split(';').collect();
                match last.len() {
                    1 => (BinSpec::Automatic, HeightSpec::Density, last[0]),
                    2 => (last[0].into(), HeightSpec::Density, last[1]),
                    _ => (BinSpec::Automatic, HeightSpec::Density, "weight"),
                }
            } else {
                (BinSpec::Automatic, HeightSpec::Density, "weight")
            };

            // convert each string to a closure

            type ParticleOutput<'a,T> = Box<dyn Fn(&T) -> f64 + 'a>;

            let funcs: Vec<Option<ParticleOutput<T>>> = ss
                .iter()
                .map(|&s| {
                    match s {
                        "x" => Some(Box::new(position) as ParticleOutput<T>),
                        "r" => Some(Box::new(radius) as ParticleOutput<T>),
                        "energy" => Some(Box::new(energy) as ParticleOutput<T>),
                        "px" => Some(Box::new(px) as ParticleOutput<T>),
                        "py" => Some(Box::new(py) as ParticleOutput<T>),
                        "pz" => Some(Box::new(pz) as ParticleOutput<T>),
                        "p_perp" => Some(Box::new(p_perp) as ParticleOutput<T>),
                        "theta" => Some(Box::new(polar_angle) as ParticleOutput<T>),
                        "phi" => Some(Box::new(azimuthal_angle) as ParticleOutput<T>),
                        "longitude" => Some(Box::new(longitude) as ParticleOutput<T>),
                        "latitude" => Some(Box::new(latitude) as ParticleOutput<T>),
                        "work" => Some(Box::new(work) as ParticleOutput<T>),
                        "chi" => Some(Box::new(chi) as ParticleOutput<T>),
                        _ => None,
                    }})
                .collect();

            let units: Vec<Option<&str>> = ss
                .iter()
                .map(|&s| {
                    match s {
                        "x" | "r" => Some("m"),
                        "energy" => Some("MeV"),
                        "px" | "py" | "pz" | "p_perp" => Some("MeV/c"),
                        "theta" | "phi" | "longitude" | "latitude" => Some("rad"),
                        "work" => Some("J"),
                        "chi" => Some("1"),
                        _ => None,
                    }})
                .collect();

            let weight_function = match weight {
                "energy" => Some(Box::new(|pt: &T| pt.energy() * pt.weight()) as ParticleOutput<T>),
                "weight" | "auto" => Some(Box::new(|pt: &T| pt.weight()) as ParticleOutput<T>),
                _ => None,
            };

            if funcs.iter().all(Option::is_some) && weight_function.is_some() {
                //println!("{:?} successfully mapped to {} funcs, bspec = {}, hspec = {}", o, funcs.len(), bspec, hspec);
                
                let (hgram, filename) = match funcs.len() {
                    1 => {
                        let hgram = Histogram::generate_1d(
                            comm, &self.store, funcs[0].as_ref().unwrap(), &weight_function.unwrap(),
                            ss[0], units[0].unwrap(), bspec, hspec
                        );

                        let mut filename = format!("!{}/{}_{}_{}", directory, index, &self.name, ss[0]);
                        if weight != "weight" {
                            filename = filename + "_" + weight;
                        }
                        if bspec == BinSpec::LogScaled {
                            filename = filename + "_log";
                        }
                        let filename = filename + ".fits";

                        (hgram, filename)
                    },
                    2 => {
                        let hgram = Histogram::generate_2d(
                            comm, &self.store,
                            [funcs[0].as_ref().unwrap(), funcs[1].as_ref().unwrap()],
                            &weight_function.unwrap(),
                            [ss[0], ss[1]], [units[0].unwrap(), units[1].unwrap()],
                            [bspec, bspec], hspec
                        );

                        let mut filename = format!("!{}/{}_{}_{}-{}", directory, index, &self.name, ss[0], ss[1]);
                        if weight != "weight" {
                            filename = filename + "_" + weight;
                        }
                        if bspec == BinSpec::LogScaled {
                            filename = filename + "_log";
                        }
                        let filename = filename + ".fits";

                        (hgram, filename)
                    }
                    _ => (None, "".to_owned())
                };

                if comm.rank() == 0 {
                    //println!("writing to {}", &filename);
                    if let Some(h) = hgram {
                        h.write_fits(&filename).map_err(|e| Error::new(ErrorKind::Other, e))?;
                    }
                }
            } // else some functions are invalid else
        } // loop over population.output

        Ok(())
    }
}



