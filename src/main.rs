use std::error::Error;
use std::path::{Path, PathBuf};

use mpi::traits::*;
use mpi::Threading;
use rand::prelude::*;
use rand_xoshiro::*;

mod constants;
use constants::*;

mod grid;
use grid::*;

mod particle;
use particle::*;

mod setup;
use setup::*;

mod qed;

fn write_energies(world: &impl Communicator, dir: &str, index: usize, grid: &impl Grid, electrons: &Population<Electron>, ions: &Population<Ion>, photons: &Population<Photon>) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let field_energy = grid.em_field_energy(world);
    let electron_energy = electrons.total_kinetic_energy(world);
    let ion_energy = ions.total_kinetic_energy(world);
    let photon_energy = photons.total_kinetic_energy(world);

    if grid.rank() == 0 {
        let filename = format!("{}/{}_energy.dat", dir, index);
        let mut file = File::create(filename)?;
        writeln! (file, "em_field {:.6e}", field_energy)?;
        writeln! (file, "electrons {:.6e}", electron_energy)?;
        writeln! (file, "ions {:.6e}", ion_energy)?;
        writeln! (file, "photons {:.6e}", photon_energy)?;
    }

    Ok(())
}

fn no_laser(_t: f64, _x: f64) -> f64 {
    0.0
}

fn main() -> Result<(), Box<dyn Error>> {
    let (universe, _) = mpi::initialize_with_threading(Threading::Funneled).unwrap();
    let world = universe.world();
    let id = world.rank();

    let mut rng = Xoshiro256StarStar::seed_from_u64(id as u64);
    // Prepare configuration file

    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or(ConfigError::raise(ConfigErrorKind::MissingFile, "", ""))?;
    let path = PathBuf::from(path);
    let output_dir = path.parent().unwrap_or(Path::new("")).to_str().unwrap_or("");

    // Read input configuration with default context

    let mut input = Config::from_file(&path)?;
    input.with_context("constants");

    let nx = input.read("control", "nx")?;
    let xmin = input.read("control", "xmin")?;
    let dx = input.read("control", "dx")?;
    let dt = 0.95 * dx / SPEED_OF_LIGHT;
    let tstart = input.read("control", "start")?;
    let tend = input.read::<f64>("control", "end")?;
    let current_deposition = input.read("control", "current_deposition")?;
    let output_frequency = input.read("control", "n_outputs")?;
    let balance = input.read("control", "balance").unwrap_or(true); // balance by default

    let photon_emission = input.read("qed", "photon_emission")?;
    let photon_absorption = input.read("qed", "photon_absorption")?;

    let photon_energy_min = input.read::<f64>("qed", "photon_energy_min").ok().map(|j| 1.0e-6 * j / ELEMENTARY_CHARGE); // convert to Option, then map joules to MeV
    let photon_angle_max = input.read("qed", "photon_angle_max").ok();
    let max_formation_length = input.read("qed", "max_formation_length").ok();
    let disable_qed_after = input.read("qed", "disable_qed_after").ok();
    let disable_absorption_after = input.read("qed", "disable_absorption_after").ok();

    // Grid initialization

    let laser_y = input.func2("laser", "Ey", ["t", "x"]);
    let laser_z = input.func2("laser", "Ez", ["t", "x"]);

    // If the laser section exists, both laser_y and laser_z
    // must be specified, or there is an input error.
    // If it does not exist, then we select periodic bcs.
    let (left_bdy, right_bdy) = if input.contains("laser") {
        laser_y.as_ref().map_err(|e| e.clone())?;
        laser_z.as_ref().map_err(|e| e.clone())?;
        (Boundary::Laser, Boundary::Absorbing)
    } else {
        (Boundary::Internal, Boundary::Internal)
    };

    let min_size = YeeGrid::min_size();

    let design = if balance {
        let ne = input.func("electrons", "ne", "x")?;
        GridDesign::balanced(world, nx, xmin, dx, min_size, left_bdy, right_bdy, &ne)
    } else {
        GridDesign::unbalanced(world, nx, xmin, dx, min_size, left_bdy, right_bdy)
    };

    let mut grid = YeeGrid::build(design);

    // Particle initialization

    let epc = input.read("electrons", "npc")?;
    let eospec = input.read("electrons", "output")?;

    let mut electrons: Population<Electron> = if epc > 0 {
        let ne = input.func("electrons", "ne", "x")?;
        let ux = input.func3("electrons", "ux", ["x", "urand", "nrand"])?;
        let uy = input.func3("electrons", "uy", ["x", "urand", "nrand"])?;
        let uz = input.func3("electrons", "uz", ["x", "urand", "nrand"])?;
        Population::new(epc, ne, ux, uy, uz, &grid, &mut rng, dt)
    } else {
        Population::new_empty()
    };

    electrons.with_output(eospec).with_name("electron");

    let ipc = input.read("ions", "npc")?;

    let mut ions: Population<Ion> = if ipc > 0 {
        let iospec = input.read("ions", "output")?;
        let ion_name: String = input.read("ions", "name")?;
        let ion_charge = input.read("ions", "Z")?;
        let ion_mass = input.read("ions", "A")?;
        let ne = input.func("ions", "ni", "x")?;
        let ux = input.func3("ions", "ux", ["x", "urand", "nrand"])?;
        let uy = input.func3("ions", "uy", ["x", "urand", "nrand"])?;
        let uz = input.func3("ions", "uz", ["x", "urand", "nrand"])?;
        let mut ions = Population::new(ipc, ne, ux, uy, uz, &grid, &mut rng, dt);
        ions.with_output(iospec)
            .with_name(&ion_name)
            .map_in_place(|pt: &mut Ion| {
                pt.with_charge_state(ion_charge, ion_charge, ion_mass);
            });
        ions
    } else {
        Population::new_empty()
    };

    // Photons only looked for if 'photon_emission' or 'photon_absorption' are on
    let mut photons: Population<Photon> = if photon_emission || photon_absorption {
        let ppc = input.read("photons", "npc")?;
        let pospec = input.read("photons", "output")?;
        let mut photons: Population<Photon> = if ppc > 0 {
            let nph = input.func("photons", "nph", "x")?;
            let ux = input.func3("photons", "ux", ["x", "urand", "nrand"])?;
            let uy = input.func3("photons", "uy", ["x", "urand", "nrand"])?;
            let uz = input.func3("photons", "uz", ["x", "urand", "nrand"])?;
            Population::new(ppc, nph, ux, uy, uz, &grid, &mut rng, dt)
        } else {
            Population::new_empty()
        };
        photons.with_output(pospec).with_name("photon");
        photons
    } else {
        Population::new_empty()
    };

    // Initial conditions

    if false { // current_deposition {
        grid.deposit(electrons.all(), dt);
        grid.deposit(ions.all(), dt);
        if left_bdy == Boundary::Laser {
            grid.synchronize(world, laser_y.as_ref().unwrap(), laser_z.as_ref().unwrap(), 0.0, dt);
        } else {
            grid.synchronize(world, &no_laser, &no_laser, 0.0, dt);
        }
        grid.initialize(world);
    }

    let mut t = tstart;
    let total_steps: usize = ((tend - tstart) / dt) as usize;
    let steps_bt_output = total_steps / output_frequency;

    if id == 0 {
        let ntasks = grid.ngrids();
        let nthreads = rayon::current_num_threads();
        println!("Running {} task{} with {} thread{} per task...", ntasks, if ntasks > 1 {"s"} else {""}, nthreads, if nthreads > 1 {"s"} else {""});
        #[cfg(feature = "extra_absorption_output")] {
            println!("[writing extra absorption data to stderr]");
        }
        #[cfg(feature = "no_beaming")] {
            println!("[neglecting angular component of photon spectrum]");
        }
        #[cfg(feature = "no_radiation_reaction")] {
            println!("[radiation reaction disabled, using classical emission rates]");
        }
        #[cfg(feature = "no_stimulated_emission")] {
            if photon_absorption {
                println!("[stimulated emission disabled, running with absorption only]");
            }
        }
        #[cfg(feature = "extra_stimulated_emission_output")] {
            println!("[writing extra stimulated emission data to stderr]");
        }
        #[cfg(feature = "immobile_photons")] {
            println!("[photon push disabled]");
        }
    }

    let runtime = std::time::Instant::now();

    for i in 0..output_frequency {
        grid.write_data(world, output_dir, i)?;
        electrons.write_data(&world, &grid, output_dir, i)?;
        ions.write_data(&world, &grid, output_dir, i)?;
        photons.write_data(&world, &grid, output_dir, i)?;
        write_energies(&world, output_dir, i, &grid, &electrons, &ions, &photons)?;

        if grid.rank() == 0 {
            if i > 0 {
                println!(
                    "Output {: >4} at t = {}, RT = {}, ETTC = {}...",
                    i, SimulationTime(t), PrettyDuration::from(runtime.elapsed()),
                    PrettyDuration::from(ettc(runtime, i * steps_bt_output, output_frequency * steps_bt_output))
                );
            } else {
                println!("Output {: >4} at t = {}...", i, SimulationTime(t));
            }
        }

        //println!("{} got {} electrons at {}", grid.rank(), electrons.size(), i);

        for _j in 0..steps_bt_output {
            //println!("{} at i = {}, j = {} [steps between output = {}]", id, i, _j, steps_bt_output);
            electrons.advance(&world, &grid, dt);
            ions.advance(&world, &grid, dt);
            #[cfg(not(feature = "immobile_photons"))] {
                photons.advance(&world, &grid, dt);
            }

            if photon_absorption {
                absorb(&mut electrons, &mut photons, &mut rng, t, dt, grid.xmin(), grid.dx(), disable_qed_after, disable_absorption_after);
            }

            if photon_emission {
                emit_radiation(&mut electrons, &mut photons, &mut rng, t, photon_energy_min, photon_angle_max, max_formation_length);
            }

            if current_deposition {
                grid.clear();
                grid.deposit(electrons.all(), dt);
                grid.deposit(ions.all(), dt);
            }

            if left_bdy == Boundary::Laser {
                grid.synchronize(world, laser_y.as_ref().unwrap(), laser_z.as_ref().unwrap(), t, dt);
            } else {
                grid.synchronize(world, &no_laser, &no_laser, t, dt);
            }
            grid.advance(dt);
            t += dt;
        }

    }

    // Output at final time
    grid.write_data(world, output_dir, output_frequency)?;
    electrons.write_data(&world, &grid, output_dir, output_frequency)?;
    ions.write_data(&world, &grid, output_dir, output_frequency)?;
    photons.write_data(&world, &grid, output_dir, output_frequency)?;
    write_energies(&world, output_dir, output_frequency, &grid, &electrons, &ions, &photons)?;

    if grid.rank() == 0 {
        println!(
            "Output {: >4} at t = {}, RT = {}",
            output_frequency, SimulationTime(t), PrettyDuration::from(runtime.elapsed())
        );
    }

    Ok(())
}
