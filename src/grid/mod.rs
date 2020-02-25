//! Representation of the electromagnetic field in the simulation domain

use mpi::traits::*;
use crate::particle::Particle;

mod yee;
pub use self::yee::*;

/// The left- and right-hand boundaries of the global simulation
/// domain, i.e. the boundaries of the leftmost and rightmost
/// subdomains, are where boundary conditions for the
/// electromagnetic field must be loaded.
/// 
/// - `Internal`: periodic boundary conditions, fields and
/// particles wrap around the grid.
/// - `Laser`: propagating electromagnetic fields are injected,
/// particles crossing the boundary are deleted.
/// - `Absorbing`: electromagnetic waves crossing the boundary
/// are strongly damped, particles are deleted.
/// - `Conducting`: fields are clamped to zero at the boundary,
/// particles are deleted.
#[derive(PartialEq,Copy,Clone)]
pub enum Boundary {
    Internal,
    Laser,
    Absorbing,
    Conducting,
}

/// Specifies how the simulation domain is to be divided among
/// the various MPI processes, without actually constructing
/// a Grid instance.
pub struct GridDesign {
    id: i32,
    numtasks: i32,
    left: Boundary,
    right: Boundary,
    xmin: f64,
    dx: f64,
    nx: Vec<usize>,
    offset: Vec<f64>,
}

/// A one-dimensional grid that stores the electromagnetic fields,
/// charges and currents, suitably discretized.
#[allow(non_snake_case)]
pub trait Grid {
    /// Constructs a new Grid based on the specified GridDesign
    fn build(geometry: GridDesign) -> Self;

    /// Returns the rank associated with the calling process.
    fn rank(&self) -> i32;

    /// Returns the number of subdomains for this particular Grid.
    fn ngrids(&self) -> i32;

    /// Rank of the process responsible for the subdomain to the left
    /// of the calling process, if it exists.
    fn to_left(&self) -> Option<i32>;

    /// Rank of the process responsible for the subdomain to the right
    /// of the calling process, if it exists.
    fn to_right(&self) -> Option<i32>;

    /// Synchronizes the fields and currents across all MPI processes,
    /// updates ghost zones, and loads boundary conditions at the
    /// specified time.
    /// 
    /// Must be called on all processes.
    fn synchronize(&mut self, comm: impl Communicator, laser_y: &impl Fn(f64, f64) -> f64, laser_z: &impl Fn(f64, f64) -> f64, t: f64, dt: f64);

    /// Calls the Maxwell solver that advance the stored electric and magnetic fields.
    fn advance(&mut self, dt: f64);

    /// Returns a tuple of (E, B), where E and B are the electric and
    /// magnetic field vectors, at cell `c`, fractional offset `x`.
    fn fields_at(&self, c: isize, x: f64) -> ([f64; 3], [f64; 3]);

    /// Coordinate of the left-hand boundary of the calling process's subdomain.
    fn xmin(&self) -> f64;

    /// Size of a single grid cell.
    fn dx(&self) -> f64;

    /// Returns the number of cells on the calling process's subdomain,
    /// not including ghost zones.
    fn size(&self) -> usize;

    /// Zeroes out the charges and currents, in preparation for a new
    /// current deposition phase.
    fn clear(&mut self);

    /// Add to the grid the current density associated with a slice of
    /// macroparticles.
    fn deposit<P: Particle + Send + Sync>(&mut self, pt: &[P], dt: f64);

    /// Optionally solve Poisson's equation for the charges and currents
    /// loaded at the beginning of the simulation, to initialize the electric
    /// and magnetic fields.
    /// 
    /// Involves synchronization, so must be called on all processes.
    fn initialize(&mut self, world: impl Communicator);

    /// Prints to file the field and currents at the current time.
    /// `dir` specifies the target directory and `index` identifies
    /// the current output.
    /// 
    /// Must be called on all processes.
    fn write_data(&self, world: impl Communicator, dir: &str, index: usize) -> std::io::Result<()>;

    /// Returns the minimum number of cells that this Grid can be
    /// constructed with.
    fn min_size() -> usize;

    /// Returns the total electromagnetic field energy (in joules)
    /// stored on the grid.
    /// 
    /// Must be called on all processes.
    fn em_field_energy(&self, world: &impl Communicator) -> f64;
}

impl GridDesign {
    /// Designs a grid which has `size` cells of dimension `dx`
    /// and is to be split among the MPI processes of `comm`.
    /// The grid is evenly divided among all processes.
    pub fn unbalanced(comm: impl Communicator, size: usize, xmin: f64, dx: f64, min_subsize: usize, left_bdy: Boundary, right_bdy: Boundary) -> Self {
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
        GridDesign {
            id: comm.rank(),
            numtasks: comm.size(),
            left: left_bdy,
            right: right_bdy,
            xmin: xmin,
            dx: dx,
            nx: nx,
            offset: offset
        }
    }

    /// Designs a grid which has `size` cells of dimension `dx`
    /// and is to be split among the MPI processes of `comm`.
    /// The split is balanced such that the number of real electrons
    /// per subdomain is approximately the same.
    pub fn balanced(comm: impl Communicator, size: usize, xmin: f64, dx: f64, min_subsize: usize, left_bdy: Boundary, right_bdy: Boundary, ne: &impl Fn(f64) -> f64) -> Self {
        let numtasks = comm.size() as usize;
        if numtasks == 0 {
            return GridDesign::unbalanced(comm, size, xmin, dx, min_subsize, left_bdy, right_bdy);
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
        GridDesign {
            id: comm.rank(),
            numtasks: comm.size(),
            left: left_bdy,
            right: right_bdy,
            xmin: xmin,
            dx: dx,
            nx: ncells,
            offset: offset
        }
    }
}