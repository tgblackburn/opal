//! Interactions that couple different particle populations.

use rayon::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;

use crate::constants::*;
use super::{Population, Particle, Electron, Photon};

pub fn emit_radiation(e: &mut Population<Electron>, ph: &mut Population<Photon>, rng: &mut Xoshiro256StarStar, t: f64, min_energy: Option<f64>, max_angle: Option<f64>, max_formation_length: Option<f64>) {
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
                let result = e.radiate(&mut rng, t);
                if result.is_none() {
                    continue;
                } else {
                    let (photon, formation_length) = result.unwrap();

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

                    let formation_length_ok = if let Some(max) = max_formation_length {
                        formation_length < max
                    } else {
                        true
                    };

                    if energy_within_bounds && angle_within_bounds && formation_length_ok {
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

#[allow(unused)]
pub fn absorb(e: &mut Population<Electron>, ph: &mut Population<Photon>, t: f64, dt: f64, xmin: f64, dx: f64, max_displacement: Option<f64>, stop_time: Option<f64>) {
    const PHOTON_E_ECRIT_CUTOFF: f64 = 1.0e-8;

    if e.store.is_empty() || ph.store.is_empty() {
        return;
    }

    let nph = ph.store.len();
    let nthreads = rayon::current_num_threads();
    let chunk_len = if nph > nthreads {
        nph / nthreads
    } else {
        nph // guarantee this runs
    };

    // return a Vec of:
    //  index of the electron that did the absorbing
    //  and the photon that was absorbed
    let absorbed: Vec<(usize,Photon)> =
        ph.store
        .par_chunks_mut(chunk_len)
        .map(
            |chunk: &mut [Photon]| -> Vec<(usize,Photon)> {
                let mut prev_cell: isize = -1;
                let mut start: Option<usize> = None;
                let mut end: Option<usize> = None;
                let mut absorbed: Vec<(usize,Photon)> = Vec::new();

                for photon in chunk.iter_mut() {
                    if photon.chi() * ELECTRON_MASS_MEV / photon.energy() < PHOTON_E_ECRIT_CUTOFF {
                        continue;
                    }

                    if let Some(t_max) = stop_time {
                        if t - photon.birth_time() > t_max {
                            continue;
                        }
                    }

                    // ignore photons that have travelled a given perp distance
                    if let Some(r_max) = max_displacement {
                        if photon.transverse_displacement() > r_max {
                            continue;
                        }
                    }

                    let (cell, _, _) = photon.location();
                    //println!("photon in cell {}", cell);

                    // find start and end indices of electrons in the same cell
                    if cell != prev_cell {
                        //println!("finding electrons in {}, starting from {}", cell, end.unwrap_or(0));
                        start = e.store.iter()
                            .skip( end.unwrap_or(0) )
                            .position(|e| e.location().0 == cell)
                            .map(|i| i + end.unwrap_or(0));

                        /*
                        if start.is_some() {
                            println!("first electron at {}, cell = {:?}", start.unwrap(), e.store[start.unwrap()].location());
                        } else {
                            println!("start is none");
                        }
                        */

                        end = if start.is_none() {
                            None
                        } else {
                            e.store.iter()
                                .skip( start.unwrap() )
                                .position(|e| e.location().0 != cell)
                                .map(|i| i + start.unwrap() )
                        };

                        /*
                        if end.is_some() {
                            println!("final electron at {}, cell = {:?}", end.unwrap() - 1, e.store[end.unwrap() - 1].location());
                        } else {
                            println!("end is none");
                        }
                        */
                    }
                    
                    prev_cell = cell;

                    // both start and end must be Some, or there are no electrons
                    // in the same cell as the present photon
                    if start.is_none() || end.is_none() {
                        //println!("No electron in same cell");
                        continue;
                    }

                    //println!("testing against electrons {}..{}", start.unwrap(), end.unwrap());
                    let electrons = &e.store[start.unwrap()..end.unwrap()];

                    for (j, e) in electrons.iter().enumerate() {
                        if photon.is_absorbed_by(e, dt, dx) {
                            photon.flag(); // mark photon for deletion
                            absorbed.push( (j + start.unwrap(), photon.clone()) );
                            break; // jump to next photon
                        }
                    } // loop over electrons in same cell
                } // loop over all photons

                absorbed
            })
        .reduce(|| Vec::<(usize,Photon)>::new(), |a, b| [a,b].concat());

    #[cfg(feature = "extra_absorption_output")] {
        if absorbed.len() > 0 {
            for (i, photon) in absorbed.iter() {
                let electron = &e.store[*i];
                let k = photon.normalized_four_momentum();
                let p = electron.normalized_four_momentum();
                let (c, xi, _) = photon.location();
                eprintln!("{:.6e} {:.6e} {:.6e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e}", xmin + dx * ((c as f64) + xi), t, photon.birth_time(), photon.chi(), k[0], k[1], k[2], k[3], electron.chi(), p[0], p[1], p[2], p[3]);
            }
        }
    }
    
    if absorbed.len() > 0 {
        //ph.store.retain(|ph| !ph.is_flagged());
        //for i in (0..ph.store.len()).rev() {
        //    if ph.store[i].is_flagged() {
        //        ph.store.swap_remove(i);
        //    }
        //}
        let mut tmp: Vec<Photon> = Vec::new();
        std::mem::swap(&mut ph.store, &mut tmp); // ph.store holds empty Vec, can consume tmp
        let mut split: Vec<Vec<Photon>> = tmp.into_par_iter().chunks(chunk_len).collect();
        split.par_iter_mut().for_each(|s: &mut Vec<Photon>| {
            for i in (0..s.len()).rev() {
                if s[i].is_flagged() {
                    s.swap_remove(i);
                }
            }
        });
        ph.store = split.concat();
    }

    // Each electron can only absorb one photon per timestep
    //absorbed.dedup_by_key(|(i, _ph)| *i);
    // handle energy change
    for (i, photon) in absorbed.iter() {
        e.store.get_mut(*i).map(|pt| pt.absorb(photon));
    }

    //if absorbed.len() > 0 {
    //    println!("Absorbed {} macrophotons, len {} -> {}", absorbed.len(), nph, ph.store.len());
    //}
}