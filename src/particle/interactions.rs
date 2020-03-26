//! Interactions that couple different particle populations.

use rayon::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;

use crate::constants::*;
use super::{Population, Particle, Electron, Photon, BinaryInteraction};

/// Implements radiation reaction, i.e. the recoil experienced by an
/// electron when it emits a photon.
/// 
/// By default, this samples the angularly resolved quantum synchrotron
/// spectrum and therefore recoil occurs stochastically.
/// These behaviours can be altered by compiling opal with
/// the features "no_beaming" (which disables the angular sampling)
/// and/or "no_radiation_reaction" (which disables the recoil and
/// switches to the classical emission spectrum).
/// 
/// # Arguments
/// - `e`: the population of radiating electrons.
/// 
/// - `ph`: the population to add newly created photons to.
/// 
/// - `rng`: in threaded mode, each thread clones the supplied rng
/// and jumps forward by different amounts; a long jump is performed
/// by the main thread before `emit_radiation` returns.
/// 
/// - `t`: photons record the time they were created, so this needs
/// to be supplied.
/// 
/// - `min_energy`: while photons of all energies are created and
/// cause recoil, they are only added to `ph` if their energy exceeds
/// this optional threshold (given in MeV).
/// 
/// - `max_angle`: similarly, it is sometimes useful to record only
/// those photons with polar angle smaller than a certain limit;
/// the angle here is measured w.r.t. the *negative* x-axis.
/// 
/// - `max_formation_length`: the formation length is estimated
/// for all photons, whether "no_beaming" is specified or not,
/// using their angle of emission and the parent electron's
/// instantaneous radius of curvature; if enabled, only photons
/// with a formation length smaller than this value are written
/// to `ph`.
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

/// Implements one-photon absorption, i.e. the time reversal of photon emission.
/// 
/// # Requirements
/// 
/// This function assumes that both populations have been sorted into
/// ascending order by cell index. This is the case if called directly
/// after the particle push.
/// 
/// The photon population will *not* be sorted afterwards, if any
/// absorption has taken place and therefore photons have been deleted.
/// The electron order is unchanged.
/// 
/// # Arguments
/// - `e`: the electron population doing the absorbing.
/// 
/// - `ph`: the population of photons that are to be absorbed.
/// 
/// - `t`: absorption can be disabled for photons after a specified interval
/// has elapsed since their creation, so the current simulation time needs to
/// be supplied.
/// 
/// - `dt`: simulation timestep.
/// 
/// - `xmin`: left-hand boundary of the local subdomain: as particles record
/// only their offset from this position, calculating the photon position
/// when "extra_absorption_output" is enabled requires this to be passed.
/// 
/// - `dx`: grid spacing, passed for the same reason as `xmin`.
/// 
/// - `max_displacement`: if enabled, disable absorption for photons that
/// have travelled a perpendicular distance greater than this value,
/// since their creation. Mimics a finite focal spot size.
/// 
/// - `stop_time`: if enabled, disable absorption for photons after
/// the specified interval has elapsed since their creation.
#[allow(unused)]
pub fn absorb(e: &mut Population<Electron>, ph: &mut Population<Photon>, rng: &mut Xoshiro256StarStar, t: f64, dt: f64, xmin: f64, dx: f64, max_displacement: Option<f64>, stop_time: Option<f64>) {
    const PHOTON_E_ECRIT_CUTOFF: f64 = 1.0e-8;

    if e.store.is_empty() || ph.store.is_empty() {
        return;
    }

    let nph = ph.store.len();
    let nthreads = rayon::current_num_threads();
    let chunk_len = (nph / (4 * nthreads)).max(1);

    // return a tuple of Vecs of:
    //  index of the electron that did the absorbing/emitting
    //  and the photon that was absorbed/stimulated
    let (absorbed, emitted) =
        ph.store
        .par_chunks_mut(chunk_len)
        .enumerate()
        .map(
            |(i, chunk): (usize, &mut [Photon])| -> (Vec<(usize,Photon)>, Vec<(usize,Photon)>) {
                let mut rng = rng.clone();
                for _ in 0..i {
                    rng.jump(); // avoid overlapping sequences of randoms
                }

                let mut prev_cell: isize = -1;
                let mut start: Option<usize> = None;
                let mut end: Option<usize> = None;
                let mut absorbed: Vec<(usize,Photon)> = Vec::new();
                let mut emitted: Vec<(usize,Photon)> = Vec::new();

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
                        let state = photon.interacts_with(e, dt, dx, &mut rng);
                        match state {
                            BinaryInteraction::PhotonAbsorbed => {
                                photon.flag(); // mark for deletion
                                absorbed.push( (j + start.unwrap(), photon.clone()) );
                                break;
                            },
                            BinaryInteraction::EmissionStimulated => {
                                emitted.push( (j + start.unwrap(), photon.with_weight(e.weight())) );
                                break;
                            },
                            _ => (), // otherwise, do nothing
                        }
                    } // loop over electrons in same cell
                } // loop over all photons

                (absorbed, emitted)
            })
        .reduce(
            || (Vec::<(usize,Photon)>::new(), Vec::<(usize,Photon)>::new()),
            |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat())
        );

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

    if emitted.len() > 0 {
        let mut extra: Vec<Photon> = emitted.iter()
            .map(|(i, ph)| ph.with_optical_depths(rng).at_time(t))
            .collect();
        // and add
        ph.store.append(&mut extra);
    }

    // Each electron can only absorb one photon per timestep
    //absorbed.dedup_by_key(|(i, _ph)| *i);
    // handle energy change
    for (i, photon) in absorbed.iter() {
        e.store.get_mut(*i).map(|pt| {
            pt.kick( photon.momentum(), photon.weight() );
        });
    }

    for (i, photon) in emitted.iter() {
        e.store.get_mut(*i).map(|pt| {
            let k = photon.momentum();
            let k = [-k[0], -k[1], -k[2]];
            pt.kick(k, pt.weight() );
        });
    }

    //if absorbed.len() > 0 {
    //    println!("Absorbed {} macrophotons, len {} -> {}", absorbed.len(), nph, ph.store.len());
    //}
    rng.long_jump();
}