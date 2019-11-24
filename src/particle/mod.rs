//#![feature(specialization)]

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
        /*
        let fields: Vec<([f64; 3], [f64; 3])> = self.store
            .par_iter()
            .map(|pt| {let (c, x) = pt.location(); grid.fields_at(c, x)})
            .collect();

        self.store.par_iter_mut()
            .zip(fields.par_iter())
            .for_each(|(ref mut pt, (E, B))| { //: (&mut T, &([f64; 3], [f64; 3]))| {
                pt.push(E, B, dx, dt);
            });
        */
        self.store.par_iter_mut() // iter_mut()
            .for_each(|pt| {
                let (c, x, _) = pt.location();
                let (E, B) = grid.fields_at(c, x);
                pt.push(&E, &B, dx, dt); 
            });

        // sort by GridCell id
        // replace with par_sort_unstable_by
        self.store.par_sort_unstable_by(
            |a, b| a.partial_cmp(b).unwrap()
        );

        // first and last elements need to be moved?
        let mut send_left: Vec<T> = Vec::new();
        let mut send_right: Vec<T> = Vec::new();

        let has_gone_left = move |pt: &&T| {
            let (c, _, _) = pt.location();
            //grid.classify_location(c) == Location::GoneLeft
            c < 0
        };

        let max = grid.size() as isize;
        let has_gone_right = move |pt: &&T| {
            let (c, _, _) = pt.location();
            //grid.classify_location(c) == Location::GoneRight
            c >= max
        };

        send_left.par_extend( // extend
            self.store.par_iter().filter(has_gone_left) // iter
        );

        send_right.par_extend(
            self.store.par_iter().filter(has_gone_right)
        );

        /*
        if send_left.len() != 0 {
            println!("{} needs to send {} pts to the left", grid.rank(), send_left.len());
        }
        if send_right.len() != 0 {
            println!("{} needs to send {} pts to the right", grid.rank(), send_right.len());
        }
        */

        // Delete lost particles

        let retain = self.store.len() - send_right.len();
        self.store.truncate(retain);
        self.store.drain(..send_left.len());

        // Even grids go first, sending to right

        let mut recv_left: Vec<T> = Vec::new();
        let mut recv_right: Vec<T> = Vec::new();

        if grid.rank() % 2 == 0 {
            if let Some(r) = grid.to_right() {
                comm.process_at_rank(r).synchronous_send(&send_right[..]);
                let mut tmp = comm.process_at_rank(r).receive_vec::<T>().0;
                recv_right.append(&mut tmp);
            }
        } else {
            if let Some(l) = grid.to_left() {
                let mut tmp = comm.process_at_rank(l).receive_vec::<T>().0;
                recv_left.append(&mut tmp);
                comm.process_at_rank(l).synchronous_send(&send_left[..]);
            }
        }

        // and then they send left

        if grid.rank() % 2 == 0 {
            if let Some(l) = grid.to_left() {
                comm.process_at_rank(l).synchronous_send(&send_left[..]);
                let mut tmp = comm.process_at_rank(l).receive_vec::<T>().0;
                recv_left.append(&mut tmp);
            }
        } else {
            if let Some(r) = grid.to_right() {
                let mut tmp = comm.process_at_rank(r).receive_vec::<T>().0;
                recv_right.append(&mut tmp);
                comm.process_at_rank(r).synchronous_send(&send_right[..]);
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
            pt.shift_cell(-(grid.size() as isize));
        }
        for pt in recv_right.iter_mut() {
            pt.shift_cell(grid.size() as isize);
        }

        self.store.extend_from_slice(&recv_left[..]);
        //self.store.splice(0..0, recv_left.into_iter());
        self.store.extend_from_slice(&recv_right[..]);
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

        for o in &self.output {
            // break into substrings, separated by colons
            let mut ss: Vec<&str> = o.split(':').collect();
            // if the final string is bracketed AND there are at least
            // two substrings, that final string might be (bspec; hspec)
            let (bspec, hspec) = if ss.len() >= 2 && ss.last().unwrap().starts_with('(') && ss.last().unwrap().ends_with(')') {
                let last = ss.pop().unwrap().trim_start_matches('(').trim_end_matches(')');
                // break this into substrings, separated by ';'
                let last: Vec<&str> = last.split(';').collect();
                match last.len() {
                    1 => (BinSpec::Automatic, last[0].into()),
                    2 => (last[0].into(), last[1].into()),
                    _ => (BinSpec::Automatic, HeightSpec::Density),
                }
            } else {
                (BinSpec::Automatic, HeightSpec::Density)
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
                        "theta" | "phi" | "theta_x" | "theta_y" => Some("rad"),
                        "work" => Some("J"),
                        "chi" => Some("1"),
                        _ => None,
                    }})
                .collect();

            if funcs.iter().all(Option::is_some) {
                //println!("{:?} successfully mapped to {} funcs, bspec = {}, hspec = {}", o, funcs.len(), bspec, hspec);
                
                let (hgram, filename) = match funcs.len() {
                    1 => {
                        let hgram = Histogram::generate_1d(
                            comm, &self.store, funcs[0].as_ref().unwrap(), &Particle::weight,
                            ss[0], units[0].unwrap(), bspec, hspec
                        );

                        let filename = if bspec == BinSpec::LogScaled {
                            format!("!{}/{}_{}_{}_log.fits", directory, index, &self.name, ss[0])
                        } else {
                            format!("!{}/{}_{}_{}.fits", directory, index, &self.name, ss[0])
                        };

                        (hgram, filename)
                    },
                    2 => {
                        let hgram = Histogram::generate_2d(
                            comm, &self.store,
                            [funcs[0].as_ref().unwrap(), funcs[1].as_ref().unwrap()],
                            &Particle::weight,
                            [ss[0], ss[1]], [units[0].unwrap(), units[1].unwrap()],
                            [bspec, bspec], hspec
                        );

                        let filename = if bspec == BinSpec::LogScaled {
                            format!("!{}/{}_{}_{}-{}_log.fits", directory, index, &self.name, ss[0], ss[1])
                        } else {
                            format!("!{}/{}_{}_{}-{}.fits", directory, index, &self.name, ss[0], ss[1])
                        };

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


