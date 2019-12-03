//#![feature(specialization)]

use std::fmt::{Debug};
use mpi::traits::*;
use rand::prelude::*;
use rand_distr::{StandardNormal, Exp1};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

mod electron;
mod photon;
mod ion;
mod vec3;
mod hgram;

// Re-export for use in main
pub use self::electron::*;
pub use self::photon::*;
pub use self::ion::*;

// For local use
use crate::grid::Grid;
use hgram::*;

#[allow(non_snake_case)]
pub trait Particle: Copy + Clone + Debug + Equivalence + PartialOrd {
    fn create(cell: isize, x: f64, u: &[f64; 3], weight: f64, dx: f64, dt: f64) -> Self;
    fn push(&mut self, E: &[f64; 3], B: &[f64; 3], dx: f64, dt: f64);
    fn location(&self) -> (isize, f64, f64);
    fn shift_cell(&mut self, delta: isize);
    fn charge(&self) -> f64;
    fn mass(&self) -> f64;
    fn velocity(&self) -> [f64; 3];
    fn energy(&self) -> f64;
    fn work(&self) -> f64;
    fn momentum(&self) -> [f64; 3];
    fn weight(&self) -> f64;
    fn chi(&self) -> f64;
    fn with_optical_depth(&self, tau: f64) -> Self;

    fn spin_state(&self) -> Option<f64> {
        None
    }
    fn spin_state_name(&self) -> Option<&'static str> {
        None
    }
}

pub struct Population<T: Particle> {
    store: Vec<T>,
    output: Vec<String>,
    name: String,
}

impl<T> Population<T> where T: Particle + Send + Sync {
    pub fn all(&self) -> &[T] {
        &self.store[..]
    }

    pub fn new_empty() -> Population<T> {
        let v: Vec<T> = Vec::new();
        Population {
            store: v,
            output: Vec::new(),
            name: String::with_capacity(16),
        }
    }

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

    pub fn map_in_place<F: Fn(&mut T)>(&mut self, f: F) -> &mut Self {
        self.store.iter_mut().for_each(f);
        self
    }

    pub fn with_output(&mut self, ospec: Vec<String>) -> &mut Self {
        self.output = ospec;
        self
    }

    pub fn with_name(&mut self, name: &str) -> &mut Self {
        self.name = name.to_owned();
        self
    }

    #[allow(non_snake_case)]
    pub fn advance<C,G>(&mut self, comm: &C, grid: &G, dt: f64)
    where C: Communicator, G: Grid + Sync {
        let dx = grid.dx();
        let max = grid.size() as isize;
        let num = self.store.len();

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
                comm.process_at_rank(r).synchronous_send(send_right);
                let mut tmp = comm.process_at_rank(r).receive_vec::<T>().0;
                recv_right.append(&mut tmp);
            }
        } else {
            if let Some(l) = grid.to_left() {
                let mut tmp = comm.process_at_rank(l).receive_vec::<T>().0;
                recv_left.append(&mut tmp);
                comm.process_at_rank(l).synchronous_send(send_left);
            }
        }

        // and then they send left

        if grid.rank() % 2 == 0 {
            if let Some(l) = grid.to_left() {
                comm.process_at_rank(l).synchronous_send(send_left);
                let mut tmp = comm.process_at_rank(l).receive_vec::<T>().0;
                recv_left.append(&mut tmp);
            }
        } else {
            if let Some(r) = grid.to_right() {
                let mut tmp = comm.process_at_rank(r).receive_vec::<T>().0;
                recv_right.append(&mut tmp);
                comm.process_at_rank(r).synchronous_send(send_right);
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

        // Need to correct cells!
        for pt in recv_left.iter_mut() {
            pt.shift_cell(-max);
        }
        for pt in recv_right.iter_mut() {
            pt.shift_cell(max);
        }

        // Overwrite lost particles with new ones
        // Append first!
        self.store.splice(num-gone_right..num, recv_right.iter().cloned());
        self.store.splice(0..gone_left, recv_left.iter().cloned()); // eat the cost, which is not so bad

        //assert!(num - gone_left - gone_right + recv_left.len() + recv_right.len() == self.store.len());
    }

    /*
    pub fn size(&self) -> usize {
        self.store.len()
    }
    */

    pub fn write_data<C,G>(&self, comm: &C, grid: &G, directory: &str, index: usize) -> std::io::Result<()> 
    where C: Communicator, G: Grid {
        use std::io::{Error, ErrorKind};
        // self.output = ["f", "f:g", "f:g:(h)", "f:g:(h;a)"]

        let position = |pt: &T| -> f64 {
            let (c, x, _) = pt.location();
            grid.xmin() + ((c as f64) + x) * grid.dx()
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
                        "energy" => Some(Box::new(energy) as ParticleOutput<T>),
                        "px" => Some(Box::new(px) as ParticleOutput<T>),
                        "py" => Some(Box::new(py) as ParticleOutput<T>),
                        "pz" => Some(Box::new(pz) as ParticleOutput<T>),
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
                        "x" => Some("m"),
                        "energy" => Some("MeV"),
                        "px" | "py" | "pz" => Some("MeV/c"),
                        "theta" | "phi" | "longitude" | "latitude" => Some("rad"),
                        "work" => Some("J"),
                        "chi" => Some("1"),
                        _ => None,
                    }})
                .collect();

            let weight_function = match weight {
                "energy" => Some(Box::new(|pt: &T| pt.energy() * pt.weight()) as ParticleOutput<T>),
                "weight" => Some(Box::new(|pt: &T| pt.weight()) as ParticleOutput<T>),
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

// Couple different particle species here:

pub fn emit_radiation(e: &mut Population<Electron>, ph: &mut Population<Photon>, rng: &mut Xoshiro256StarStar, min_energy: Option<f64>, max_angle: Option<f64>) {
    let ne = e.store.len();
    let nthreads = rayon::current_num_threads();
    // chunk length cannot be zero
    let chunk_len = if ne > nthreads {
        ne / nthreads
    } else {
        1 // guarantee this runs
    };

    // find an rng that can be skipped ahead
    // each thread clones and jumps ahead by a different amount
    // finally, rng is jumped ahead by the total
    let emitted: Vec<Photon> = e.store
        .par_chunks_mut(chunk_len)
        .enumerate()
        .map(|(i, chunk): (usize, &mut [Electron])| -> Vec<Photon> {
            let mut rng = rng.clone();
            for _ in 0..i {
                rng.jump(); // avoid overlapping sequences of randoms
            }
            let mut v: Vec<Photon> = Vec::new();
            for e in chunk {
                let photon = e.radiate(&mut rng);
                if photon.is_none() {
                    continue;
                } else {
                    let photon = photon.unwrap();

                    let energy_within_bounds = if let Some(min) = min_energy {
                        photon.energy() >= min
                    } else {
                        true
                    };

                    let angle_within_bounds = if let Some(max) = max_angle {
                        let p = photon.momentum();
                        let magnitude = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
                        let angle = f64::acos(-p[0] / magnitude); // polar angle to negative x-axis
                        angle <= max
                    } else {
                        true
                    };

                    if energy_within_bounds && angle_within_bounds {
                        v.push(photon);
                    }
                }
            }
            v
        })
        .flatten()
        .collect();

    rng.long_jump();
    ph.store.extend_from_slice(&emitted[..]);
}


