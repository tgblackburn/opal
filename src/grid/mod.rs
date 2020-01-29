use mpi::traits::*;
use crate::particle::Particle;

mod yee;
pub use self::yee::*;

#[derive(PartialEq)]
pub enum Boundary {
    Internal,
    Laser,
    Absorbing,
}

#[allow(non_snake_case)]
pub trait Grid {
    // laser has to be Sync so that Grid can be shared among threads for particle advance
    fn new(comm: impl Communicator, geometry: Geometry, left: Boundary) -> Self;
    fn rank(&self) -> i32;
    fn ngrids(&self) -> i32;
    fn to_left(&self) -> Option<i32>;
    fn to_right(&self) -> Option<i32>;
    fn synchronize(&mut self, comm: impl Communicator, laser_y: &impl Fn(f64, f64) -> f64, laser_z: &impl Fn(f64, f64) -> f64, t: f64);
    fn advance(&mut self, dt: f64);
    fn fields_at(&self, c: isize, x: f64) -> ([f64; 3], [f64; 3]);
    fn xmin(&self) -> f64;
    fn dx(&self) -> f64;
    fn size(&self) -> usize;
    fn weight(x: f64) -> f64;
    fn flux(x_i: f64, x_f: f64) -> f64;
    //fn classify_location(&self, c: isize) -> Location;
    fn clear(&mut self);
    fn deposit<P: Particle + Send + Sync>(&mut self, pt: &[P], dt: f64);
    fn initialize(&mut self, world: impl Communicator);
    fn write_data(&self, world: impl Communicator, dir: &str, index: usize) -> std::io::Result<()>;
    fn min_size() -> usize;
    fn em_field_energy(&self, world: &impl Communicator) -> f64;
}

pub struct Geometry {
    xmin: f64,
    dx: f64,
    nx: Vec<usize>,
    offset: Vec<f64>,
}

impl Geometry {
    pub fn unbalanced(comm: impl Communicator, size: usize, xmin: f64, dx: f64, min_subsize: usize) -> Self {
        let numtasks = comm.size() as usize;
        let subsize = (size / numtasks).max(min_subsize);
        let nx = vec![subsize; numtasks];

        let offset: Vec<f64> = nx
            .iter()
            .scan(0usize, |rt, n| -> Option<f64> {
                let offset = (*rt as f64) * dx;
                *rt = *rt + n;
                Some(offset)
            })
            .collect();

        //println!("nx = {:?}, offset = {:?}", nx, offset);
        Geometry {xmin: xmin, dx: dx, nx: nx, offset: offset}
    }

    pub fn balanced(comm: impl Communicator, size: usize, xmin: f64, dx: f64, min_subsize: usize, ne: &impl Fn(f64) -> f64) -> Self {
        let numtasks = comm.size() as usize;
        if numtasks == 0 {
            return Geometry::unbalanced(comm, size, xmin, dx, min_subsize);
        }

        // Get total number of macroparticles
        let ppc: Vec<f64> = (0..(size-min_subsize)) // make sure last processor gets min_subsize cells
            .map(|n: usize| {let x = xmin + (n as f64) * dx; dx * ne(x)})
            .collect();
        let cumsum: Vec<f64> = ppc.iter()
            .scan(0.0f64, |total, np| {*total += np; Some(*total)})
            .collect();
        // Each processor should get this many, approx:
        let target = cumsum.last().unwrap() / (numtasks as f64);
        // Find indexes at which to partition:
        let mut ncells: Vec<usize> = Vec::new();
        let mut start: usize = 0;
        for p in 1..numtasks {
            let i = cumsum.iter()
                .skip(start + min_subsize) // each processor gets at least six cells
                .position(|&cs| cs >= (target * (p as f64))) // find first cell past target
                .unwrap();
            ncells.push(i + min_subsize);
            start += i + min_subsize; // skipping resets itr count
        }
        let ndone = ncells.iter().sum::<usize>();
        ncells.push(size - ndone);

        let offset: Vec<f64> = ncells
            .iter()
            .scan(0usize, |rt, n| -> Option<f64> {
                let offset = (*rt as f64) * dx;
                *rt = *rt + n;
                Some(offset)
            })
            .collect();

        //println!("ncells = {:?}, offset = {:?}", ncells, offset);
        Geometry {xmin: xmin, dx: dx, nx: ncells, offset: offset}
    }
}