use std::error::Error;
use std::path::{Path, PathBuf};

use mpi::traits::*;
use mpi::Threading;
use indicatif::FormattedDuration;
use rand::prelude::*;
//use rand_chacha::*;
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

#[rustversion::since(1.38)]
fn ettc (start: std::time::Instant, current: usize, total: usize) -> std::time::Duration {
    let rt = start.elapsed().as_secs_f64();
    let ettc = rt * ((total - current) as f64) / (current as f64);
    std::time::Duration::from_secs_f64(ettc)
}

#[rustversion::before(1.38)]
fn ettc (start: std::time::Instant, current: usize, total: usize) -> std::time::Duration {
    let rt = start.elapsed();
    let rt = (rt.as_secs() as f64) + (rt.subsec_nanos() as f64) * 1.0e-9;
    let ettc = rt * ((total - current) as f64) / (current as f64);
    std::time::Duration::from_secs(ettc as u64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (universe, _) = mpi::initialize_with_threading(Threading::Funneled).unwrap();
    let world = universe.world();
    let id = world.rank();

    let mut rng = Xoshiro256StarStar::seed_from_u64(id as u64);
    qed::photon_absorption::disable_gsl_abort_on_error();

    // Prepare configuration file

    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or(InputError::InvalidInputFile("no file supplied"))?;
    let path = PathBuf::from(path);
    let output_dir = path.parent().unwrap_or(Path::new("")).to_str().unwrap_or("");

    // Read input configuration with default context

    let mut input = Configuration::from_file(&path)?;
    input.with_context("constants");

    let nx = input.integer("control", "nx")? as usize;
    let xmin = input.real("control", "xmin")?;
    let dx = input.real("control", "dx")?;
    let dt = 0.95 * dx / SPEED_OF_LIGHT;
    let tstart = input.real("control", "start")?;
    let tend = input.real("control", "end")?;
    let current_deposition = input.bool("control", "current_deposition")?;
    let output_frequency = input.integer("control", "n_outputs")? as usize;

    let photon_emission = input.bool("qed", "photon_emission")?;
    let photon_energy_min = input.real("qed", "photon_energy_min").ok(); // convert to Option
    let photon_angle_max = input.real("qed", "photon_angle_max").ok();

    let photon_absorption = input.bool("qed", "photon_absorption")?;

    // Grid initialization

    let laser = input.func2("laser", "field", ["t", "x"])?;
    let mut grid = YeeGrid::new(world, nx, xmin, dx, Boundary::Laser);

    // Particle initialization

    let epc = input.integer("electrons", "npc")?;
    let eospec = input.strings("electrons", "output")?;

    let mut electrons: Population<Electron> = if epc > 0 {
        let ne = input.func("electrons", "ne", "x")?;
        let ux = input.func3("electrons", "ux", ["x", "urand", "nrand"])?;
        let uy = input.func3("electrons", "uy", ["x", "urand", "nrand"])?;
        let uz = input.func3("electrons", "uz", ["x", "urand", "nrand"])?;
        Population::new(epc as usize, ne, ux, uy, uz, &grid, &mut rng, dt)
    } else {
        Population::new_empty()
    };

    electrons.with_output(eospec).with_name("electrons");

    let ipc = input.integer("ions", "npc")?;

    let mut ions: Population<Ion> = if ipc > 0 {
        let iospec = input.strings("ions", "output")?;
        let ion_name = input.string("ions", "name")?;
        let ion_charge = input.real("ions", "Z")?;
        let ion_mass = input.real("ions", "A")?;
        let ne = input.func("ions", "ni", "x")?;
        let ux = input.func3("ions", "ux", ["x", "urand", "nrand"])?;
        let uy = input.func3("ions", "uy", ["x", "urand", "nrand"])?;
        let uz = input.func3("ions", "uz", ["x", "urand", "nrand"])?;
        let mut ions = Population::new(ipc as usize, ne, ux, uy, uz, &grid, &mut rng, dt);
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
        let ppc = input.integer("photons", "npc")?;
        let pospec = input.strings("photons", "output")?;
        let mut photons: Population<Photon> = if ppc > 0 {
            let nph = input.func("photons", "nph", "x")?;
            let ux = input.func3("photons", "ux", ["x", "urand", "nrand"])?;
            let uy = input.func3("photons", "uy", ["x", "urand", "nrand"])?;
            let uz = input.func3("photons", "uz", ["x", "urand", "nrand"])?;
            Population::new(ppc as usize, nph, ux, uy, uz, &grid, &mut rng, dt)
        } else {
            Population::new_empty()
        };
        photons.with_output(pospec).with_name("photons");
        photons
    } else {
        Population::new_empty()
    };

    // Initial conditions

    if current_deposition {
        grid.deposit(electrons.all(), dt);
        grid.deposit(ions.all(), dt);
        grid.synchronize(world, &laser, 0.0);
        grid.initialize(world);
    }

    let mut t = tstart;
    let total_steps: usize = ((tend - tstart) / dt) as usize;
    let steps_bt_output = total_steps / output_frequency;

    println! ("Running with on {} nodes with {} threads per node...", grid.ngrids(), rayon::current_num_threads());

    let runtime = std::time::Instant::now();

    for i in 0..output_frequency {
        grid.write_data(world, output_dir, i)?;
        electrons.write_data(&world, &grid, output_dir, i)?;
        ions.write_data(&world, &grid, output_dir, i)?;
        photons.write_data(&world, &grid, output_dir, i)?;

        if grid.rank() == 0 {
            if i > 0 {
                println!(
                    "Output {: >4} at t = {: >8.2} fs, RT = {}, ETTC = {}...",
                    i, 1.0e15 * t, FormattedDuration(runtime.elapsed()),
                    FormattedDuration(ettc(runtime, i * steps_bt_output, output_frequency * steps_bt_output))
                );
            } else {
                println!("Output {: >4} at t = {: >8.2} fs...", i, 1.0e15 * t);
            }
        }

        //println!("{} got {} electrons at {}", grid.rank(), electrons.size(), i);

        for _j in 0..steps_bt_output {
            //println!("{} at i = {}, j = {} [steps between output = {}]", id, i, _j, steps_bt_output);
            electrons.advance(&world, &grid, dt);
            ions.advance(&world, &grid, dt);
            photons.advance(&world, &grid, dt);

            if photon_absorption {
                absorb(&mut electrons, &mut photons, dt, grid.dx());
            }

            if photon_emission {
                emit_radiation(&mut electrons, &mut photons, &mut rng, photon_energy_min, photon_angle_max);
            }

            if current_deposition {
                grid.clear();
                grid.deposit(electrons.all(), dt);
                grid.deposit(ions.all(), dt);
            }

            grid.synchronize(world, &laser, t);
            grid.advance(dt);
            t += dt;
        }

    }

    // Output at final time
    grid.write_data(world, output_dir, output_frequency)?;
    electrons.write_data(&world, &grid, output_dir, output_frequency)?;
    ions.write_data(&world, &grid, output_dir, output_frequency)?;
    photons.write_data(&world, &grid, output_dir, output_frequency)?;

    if grid.rank() == 0 {
        println!(
            "Output {: >4} at t = {: >8.2} fs, RT = {}",
            output_frequency, 1.0e15 * t, FormattedDuration(runtime.elapsed())
        );
    }

    Ok(())
}
