use std::error::Error;
use std::fs::File;
//use std::f64::consts;
use std::fmt;
use std::path::{Path, PathBuf};

use mpi::traits::*;
use indicatif::FormattedDuration;
use yaml_rust::{yaml::Yaml, YamlLoader};
use rand::prelude::*;
use rand_chacha::*;

mod constants;
use constants::*;

mod grid;
use grid::*;

mod particle;
use particle::*;

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

enum InputError {
    InvalidInputFile(&'static str),
    CouldNotParse(String, String),
    MissingField(String),
}

impl fmt::Debug for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use InputError::*;
        let help_msg = "Usage: mpirun -n np ./opal input-file";
        match self {
            InvalidInputFile(s) => write!(f, "invalid input file: {}\n{}", s, help_msg),
            CouldNotParse(token,field) => write!(f, "unable to parse '{}' = '{}' in configuration file", token, field),
            MissingField(token) => write!(f, "unable to find '{}' with correct type in configuration file", token),
        }
    }
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for InputError {}

fn read_evaluate<C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, ctx: &C) -> Result<f64,InputError> {
    let name = field.to_owned();
    match &input[section][field] {
        Yaml::Real(s) => s.parse::<f64>().map_err(|_| InputError::CouldNotParse(name.clone(), s.clone())),
        //Yaml::String(s) => s.parse::<meval::Expr>()?.eval_with_context(default_ctx),
        Yaml::String(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(name.clone(), s.clone()))?; // Result<f64,meval:;err>
            expr.eval_with_context(ctx).map_err(|_| InputError::CouldNotParse(name.clone(), s.clone()))
        },
        _ => Err(InputError::MissingField(name)),
    }  
}

fn read_func1<'a, C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, arg: &str, ctx: &'a C) -> Result<impl Fn(f64) -> f64 + 'a,InputError> {
    match &input[section][field] {
        Yaml::String(s) | Yaml::Real(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
            expr.bind_with_context(ctx, arg).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
        },
        _ => Err(InputError::MissingField(field.to_owned()))
    }
}

fn read_func2<'a, C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, args: [&str; 2], ctx: &'a C) -> Result<impl Fn(f64, f64) -> f64 + 'a,InputError> {
    match &input[section][field] {
        Yaml::String(s) | Yaml::Real(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
            expr.bind2_with_context(ctx, args[0], args[1]).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
        },
        _ => Err(InputError::MissingField(field.to_owned()))
    }
}

fn read_func3<'a, C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, args: [&str; 3], ctx: &'a C) -> Result<impl Fn(f64, f64, f64) -> f64 + 'a,InputError> {
    match &input[section][field] {
        Yaml::String(s) | Yaml::Real(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
            expr.bind3_with_context(ctx, args[0], args[1], args[2]).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
        },
        _ => Err(InputError::MissingField(field.to_owned()))
    }
}

fn read_integer(input: &Yaml, section: &str, field: &str) -> Result<i64,InputError> {
    match &input[section][field] {
        Yaml::Integer(i) => Ok(*i),
        _ => Err(InputError::MissingField(field.to_owned())),
    }
}

fn read_bool(input: &Yaml, section: &str, field: &str) -> Result<bool,InputError> {
    match &input[section][field] {
        Yaml::Boolean(b) => Ok(*b),
        _ => Err(InputError::MissingField(field.to_owned())),
    }
}

fn read_strings(input: &Yaml, section: &str, field: &str) -> Result<Vec<String>, InputError> {
    let name = field.to_owned();
    match &input[section][field] {
        Yaml::String(s) => {
            Ok(vec![s.clone()])
        },
        Yaml::Array(array) => {
            // a is a vec of Vec<Yaml>
            let take_yaml_string = |y: &Yaml| -> Option<String> {
                match y {
                    Yaml::String(s) => Some(s.clone()),
                    _ => None
                }
            };
            let got: Vec<String> = array.iter().filter_map(take_yaml_string).collect();
            if got.is_empty() {
                Err(InputError::CouldNotParse(section.to_owned(), name))
            } else {
                Ok(got)
            }
        },
        _ => Err(InputError::MissingField(name))
    }
}

fn read_string(input: &Yaml, section: &str, field: &str) -> Result<String, InputError> {
    let strs = read_strings(input, section, field)?;
    //let str = strs.first().ok_or(InputError::MissingField(field.to_owned()))?;
    //str.clone()
    Ok(strs[0].clone())
}

// return a result type for error handling
fn read_to_context(input: &Yaml, section: &str, ctx: &mut meval::Context) {
    let tmp = ctx.clone();
    //println!("{:?}", input["constants"].as_hash());
    for (a, b) in input[section].as_hash().unwrap() {
        //println!("{:?} {:?}", a, b);
        match (a, b) {
            (Yaml::String(s), Yaml::Real(v)) => {
                if let Ok(num) = v.parse::<f64>() {ctx.var(s, num);}
            },
            (Yaml::String(s), Yaml::String(v)) => {
                if let Ok(expr) = v.parse::<meval::Expr>() {
                    if let Ok(num) = expr.eval_with_context(&tmp) {ctx.var(s, num);}
                }
            },
            _ => ()
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let id = world.rank();

    let mut rng = ChaCha8Rng::seed_from_u64(id as u64);

    // Prepare configuration file

    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or(InputError::InvalidInputFile("no file supplied"))?;
    let path = PathBuf::from(path);

    let input = std::fs::read_to_string(&path)?;
    let input = YamlLoader::load_from_str(&input)?;
    let input = input.first().ok_or(InputError::InvalidInputFile("yaml trouble"))?;
    let output_dir = path.parent().unwrap_or(Path::new("")).to_str().unwrap_or("");

    // Read input configuration with default context

    let mut default_ctx = meval::Context::new();
    default_ctx
        .var("m", ELECTRON_MASS)
        .var("me", ELECTRON_MASS)
        .var("mp", PROTON_MASS)
        .var("c", SPEED_OF_LIGHT)
        .var("e", ELEMENTARY_CHARGE)
        .var("eV", ELEMENTARY_CHARGE)
        .var("keV", 1.0e3 * ELEMENTARY_CHARGE)
        .var("MeV", 1.0e6 * ELEMENTARY_CHARGE)
        .var("micro", 1.0e-6)
        .func3("step", |x, min, max| if x >= min && x < max {1.0} else {0.0})
        .func("critical", |omega| VACUUM_PERMITTIVITY * ELECTRON_MASS * omega.powi(2) / ELEMENTARY_CHARGE.powi(2));

    read_to_context(input, "constants", &mut default_ctx);

    let nx = read_integer(input, "control", "nx")? as usize;
    let xmin = read_evaluate(input, "control", "xmin", &default_ctx)?;
    let dx = read_evaluate(input, "control", "dx", &default_ctx)?;
    let dt = 0.95 * dx / SPEED_OF_LIGHT;
    let tstart = read_evaluate(input, "control", "start", &default_ctx)?;
    let tend = read_evaluate(input, "control", "end", &default_ctx)?;
    let current_deposition = read_bool(input, "control", "current_deposition")?;
    let output_frequency = read_integer(input, "control", "n_outputs")? as usize;

    // Grid initialization

    let laser = read_func2(input, "laser", "field", ["t", "x"], &default_ctx)?;
    let mut grid = YeeGrid::new(world, nx, xmin, dx, Boundary::Laser);

    // Particle initialization

    let epc = read_integer(input, "electrons", "npc")?;
    let eospec = read_strings(input, "electrons", "output")?;

    let mut electrons: Population<Electron> = if epc > 0 {
        let ne = read_func1(input, "electrons", "ne", "x", &default_ctx)?;
        let ux = read_func3(input, "electrons", "ux", ["x", "urand", "nrand"], &default_ctx)?;
        let uy = read_func3(input, "electrons", "uy", ["x", "urand", "nrand"], &default_ctx)?;
        let uz = read_func3(input, "electrons", "uz", ["x", "urand", "nrand"], &default_ctx)?;
        Population::new(epc as usize, ne, ux, uy, uz, &grid, &mut rng, dt)
    } else {
        Population::new_empty()
    };

    electrons.with_output(eospec).with_name("electrons");

    let ipc = read_integer(input, "ions", "npc")?;

    let mut ions: Population<Ion> = if ipc > 0 {
        let iospec = read_strings(input, "ions", "output")?;
        let ion_name = read_string(input, "ions", "name")?;
        let ion_charge = read_evaluate(input, "ions", "Z", &default_ctx)?;
        let ion_mass = read_evaluate(input, "ions", "A", &default_ctx)?;
        let ne = read_func1(input, "ions", "ni", "x", &default_ctx)?;
        let ux = read_func3(input, "ions", "ux", ["x", "urand", "nrand"], &default_ctx)?;
        let uy = read_func3(input, "ions", "uy", ["x", "urand", "nrand"], &default_ctx)?;
        let uz = read_func3(input, "ions", "uz", ["x", "urand", "nrand"], &default_ctx)?;
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

    // Initial conditions

    if current_deposition {
        grid.deposit(electrons.all(), dt);
        grid.deposit(ions.all(), dt);
        grid.synchronize(world, &laser, 0.0);
        grid.initialize(world);
    }

    let mut t = tstart;
    let total_steps: usize = ((tend - tstart) / dt) as usize;
    //let output_frequency: usize = 12;
    let steps_bt_output = total_steps / output_frequency;

    println! ("Running with on {} nodes with {} threads per node...", grid.ngrids(), rayon::current_num_threads());

    let runtime = std::time::Instant::now();

    for i in 0..output_frequency {
        let mut file = File::create( format!("{}/grid_{}.dat", output_dir, i) )?;
        grid.write(world, &mut file)?;

        electrons.write_data(&world, &grid, output_dir, i)?;
        ions.write_data(&world, &grid, output_dir, i)?;

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
            electrons.advance(&world, &grid, dt);
            ions.advance(&world, &grid, dt);

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
    electrons.write_data(&world, &grid, output_dir, output_frequency)?;
    ions.write_data(&world, &grid, output_dir, output_frequency)?;
    if grid.rank() == 0 {
        println!(
            "Output {: >4} at t = {: >8.2} fs, RT = {}",
            output_frequency, 1.0e15 * t, FormattedDuration(runtime.elapsed())
        );
    }

    Ok(())
}
