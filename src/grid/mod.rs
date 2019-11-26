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
    fn new(comm: impl Communicator, size: usize, x: f64, dx: f64, left: Boundary) -> Self;
    fn rank(&self) -> i32;
    fn ngrids(&self) -> i32;
    fn to_left(&self) -> Option<i32>;
    fn to_right(&self) -> Option<i32>;
    fn synchronize(&mut self, comm: impl Communicator, laser: &impl Fn(f64, f64) -> f64, t: f64);
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
}
