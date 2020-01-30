use std::error::Error;
use std::path::{Path, PathBuf};
use std::fmt;

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

struct PrettyDuration {
    pub duration: std::time::Duration,
}

impl From<std::time::Duration> for PrettyDuration {
    fn from(duration: std::time::Duration) -> PrettyDuration {
        PrettyDuration {duration: duration}
    }
}

impl fmt::Display for PrettyDuration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut t = self.duration.as_secs();
        let s = t % 60;
        t /= 60;
        let min = t % 60;
        t /= 60;
        let hr = t % 24;
        let d = t / 24;
        if d > 0 {
            write!(f, "{}d {:02}:{:02}:{:02}", d, hr, min, s)
        } else {
            write!(f, "{:02}:{:02}:{:02}", hr, min, s)
        }
    }
}

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
    let balance = input.bool("control", "balance").unwrap_or(true); // balance by default

    let photon_emission = input.bool("qed", "photon_emission")?;
    let photon_energy_min = input.real("qed", "photon_energy_min").ok().map(|j| 1.0e-6 * j / ELEMENTARY_CHARGE); // convert to Option, then map joules to MeV
    let photon_angle_max = input.real("qed", "photon_angle_max").ok();
    let max_formation_length = input.real("qed", "max_formation_length").ok();
    let disable_qed_after = input.real("qed", "disable_qed_after").ok();

    let photon_absorption = input.bool("qed", "photon_absorption")?;

    // Grid initialization

    let laser_y = input.func2("laser", "Ey", ["t", "x"])?;
    let laser_z = input.func2("laser", "Ez", ["t", "x"])?;
    let min_size = YeeGrid::min_size();

    let geometry = if balance {
        let ne = input.func("electrons", "ne", "x")?;
        Geometry::balanced(world, nx, xmin, dx, min_size, &ne)
    } else {
        Geometry::unbalanced(world, nx, xmin, dx, min_size)
    };

    let mut grid = YeeGrid::new(world, geometry, Boundary::Laser);

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

    if false { // current_deposition {
        grid.deposit(electrons.all(), dt);
        grid.deposit(ions.all(), dt);
        grid.synchronize(world, &laser_y, &laser_z, 0.0);
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
                    "Output {: >4} at t = {: >8.2} fs, RT = {}, ETTC = {}...",
                    i, 1.0e15 * t, PrettyDuration::from(runtime.elapsed()),
                    PrettyDuration::from(ettc(runtime, i * steps_bt_output, output_frequency * steps_bt_output))
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
                absorb(&mut electrons, &mut photons, t, dt, grid.xmin(),grid.dx(), disable_qed_after);
            }

            if photon_emission {
                emit_radiation(&mut electrons, &mut photons, &mut rng, t, photon_energy_min, photon_angle_max, max_formation_length);
            }

            if current_deposition {
                grid.clear();
                grid.deposit(electrons.all(), dt);
                grid.deposit(ions.all(), dt);
            }

            grid.synchronize(world, &laser_y, &laser_z, t);
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
            "Output {: >4} at t = {: >8.2} fs, RT = {}",
            output_frequency, 1.0e15 * t, PrettyDuration::from(runtime.elapsed())
        );
    }

    Ok(())
}
