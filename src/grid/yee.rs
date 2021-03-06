//! Module implementing YeeGrid and its associated Cell.
//!
//! Call in this order:
//!   - output (E, B and particle positions at t = 0)
//!   - map to particle, push
//!   - deposit currents, synchronize grid
//!   - half B advance
//!   - full E advance
//!   - half B advance
//!
//! Why? These are the times at which quantities are stored on the grid.
//! For the field-mapping stage, we need only the cells [-1] to [n+1].
//! ```text 
//!    | [-4]  [-3]  [-2]  [-1]  |  [0]           [n-1] |  [n]  [n+1] [n+2] [n+3] |
//!  --|-------------------------|------- - - - --------|-------------------------|
//!  J |                         |                      |                         |
//!  B |                     0   |   0              0   |   0     0               |
//!  E |                     0   |   0              0   |   0     0               |
//! ```
//! The push advances positions from 0 to 1, and velocities from -1/2 to 1/2.
//! Following push, current deposition and synchronization, everything is aligned
//! ```text
//!    | [-4]  [-3]  [-2]  [-1]  |  [0]           [n-1] |  [n]  [n+1] [n+2] [n+3] |
//!  --|-------------------------|------- - - - --------|-------------------------|
//!  J |  1/2   1/2   1/2   1/2  |                      |  1/2   1/2   1/2   1/2  |
//!  B |   0     0     0     0   |   0              0   |   0     0     0     0   |
//!  E |   0     0     0     0   |   0              0   |   0     0     0     0   |
//! ```
//! Advance B by half a step in all cells [-4] to [n+2]. This requires E at t = 0
//! locally and in the cell to the right:
//! ```text
//!    | [-4]  [-3]  [-2]  [-1]  |  [0]           [n-1] |  [n]  [n+1] [n+2] [n+3] |
//!  --|-------------------------|------- - - - --------|-------------------------|
//!  J |  1/2   1/2   1/2   1/2  |                      |  1/2   1/2   1/2   1/2  |
//!  B |  1/2   1/2   1/2   1/2  |  1/2            1/2  |  1/2   1/2   1/2    0   |
//!  E |   0     0     0     0   |   0              0   |   0     0     0     0   |
//! ```
//! Then advance E by a full step in cells [-3] to [n+3]. This requires j and B
//! at t = 1/2 locally and in the cell to the left. An error is made in [n+3].
//! Silver-Muller bc applied in [-4].
//! ```text
//!    | [-4]  [-3]  [-2]  [-1]  |  [0]           [n-1] |  [n]  [n+1] [n+2] [n+3] |
//!  --|-------------------------|------- - - - --------|-------------------------|
//!  J |  1/2   1/2   1/2   1/2  |                      |  1/2   1/2   1/2   1/2  |
//!  B |  1/2   1/2   1/2   1/2  |  1/2            1/2  |  1/2   1/2   1/2    0   |
//!  E |   1     1     1     1   |   1              1   |   1     1     1     E   |
//! ```
//! Now finish the B advance in cells [-4] to [n+2]. This requires E at t = 1
//! locally and in the cell to the right.
//! ```text
//!    | [-4]  [-3]  [-2]  [-1]  |  [0]           [n-1] |  [n]  [n+1] [n+2] [n+3] |
//!  --|-------------------------|------- - - - --------|-------------------------|
//!  J |  1/2   1/2   1/2   1/2  |                      |  1/2   1/2   1/2   1/2  |
//!  B |   1     1     1     1   |   1              1   |   1     1     E     0   |
//!  E |   1     1     1     1   |   1              1   |   1     1     1     E   |
//! ```
//! Cells containing E are incorrect, but when the loop repeats, these are
//! overwritten by the synchronization.

use std::ops::Add;
use mpi::traits::*;
use mpi::datatype::UserDatatype;
use memoffset::*;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::constants::*;
use crate::grid::*;

/// Cell represents a small spatial region of the specified grid,
/// over which field and current quantities are considered to be constant.
///
/// YeeGrid stores a 1D array of these cells, so that the nxth cell
/// has this structure, where nx is in the range [0, grid.size - 1]:
/// ```text
///        nx * dx          (nx + 1) * dx
///           |                   |
///           |---------|---------|
///           |                   |
/// ```
/// rho, jy, jz, Ey, Ez and Bx are defined at the left-hand edge.
/// jx, Ex, By and Bz are defined at the cell centre.
#[allow(non_snake_case)]
#[derive(Copy,Clone)]
#[repr(C)]
struct Cell {
    pub x: f64,
    pub rho: f64,
    pub j: [f64; 3],
    pub E: [f64; 3],
    pub B: [f64; 3],
}

impl Cell {
    /// Adds to `self` the charges and currents in `other`,
    /// and *overwrites* E and B.
    fn overlay_ghost(&mut self, other: &Cell) {
        self.rho += other.rho;
        self.j[0] += other.j[0];
        self.j[1] += other.j[1];
        self.j[2] += other.j[2];
        self.E = other.E;
        self.B = other.B;
    }

    /// Adds to `self` the charges and currents in `other`,
    /// but leaves E and B unchanged.
    fn overlay(&mut self, other: &Cell) {
        self.rho += other.rho;
        self.j[0] += other.j[0];
        self.j[1] += other.j[1];
        self.j[2] += other.j[2];
    }

    /// Weighting function to map grid point values to the particle.
    /// ```text
    ///                         xhat
    ///                   |<------------>|
    ///                 1 .       .------|------.
    ///                 .'|`.     |      |      |
    ///               .'  |  `.   |      |      |
    ///             .'    |    `. |      |      |
    ///           .'      |      `.      |      |
    ///         .'        |       |`.    |      |
    ///       .'          |       |XX`.  |      |
    /// <----|------------|-------|----|-|------|----> x / dx
    ///      -1           0            1
    /// ```
    /// xhat is the distance from the particle centre to a point on the grid, measured
    /// in units of the grid spacing. The weight associated with that point is given
    /// by weight(xhat), which is non-zero for 0 < xhat < 3/2. The sum of all the weights
    /// of grid points within 3/2 of the particle centre is 1.
    ///
    /// Grid points have shape defined by a b-spline of order 0 i.e. a top-hat of
    ///total width dx. Particles have shape defined by a b-spline of order 1, i.e.
    /// a triangle with total width 2 dx.
    ///
    /// By the definition of b-splines, the interpolation function, i.e. the
    /// convolution of these shapes, is a b-spline of order 2.
    fn weight(xi: f64) -> f64 {
        let xhat = xi.abs();
        if xhat > 1.5 {
            return 0.0;
        } else if xhat < 0.5 {
            return 0.75 - xhat.powi(2);
        } else {
            return 1.125 - 1.5 * xhat + 0.5 * xhat.powi(2);
        }
    }

    /// flux gives the `amount' of particle that has flowed through an
    /// imaginary boundary whose displacement from the particle centre is initially
    /// x_i and finally x_f.
    ///
    /// The magnitude of that flux is given by the shaded region pictured here:
    /// ```text
    ///                    |
    ///                    .
    ///                  .'|`. |<--| movement of boundary
    ///                .'  |  `.
    ///              .'    |   |`.
    ///            .'      |   |XX`.
    ///          .'        |   |XXX|`.
    ///        .'          |   |XXX|  `.
    ///  <----|------------|---|---|----|----> x / dx
    ///       -1           0   |   |    1
    ///                       x_f x_i
    /// ```
    /// NB: We assume that the boundary moves less than dx i.e. x_i - x_f < dx
    ///
    /// We use the following sign conventions:
    ///
    /// x := x_bdy - x_pt i.e. if the particle is to the *left* of the boundary
    ///                       the initial distance is considered positive.
    ///
    /// If x_i > x_f, the particle is moving across the boundary from left to right
    /// and the flux is considered positive.
    ///
    /// This function exactly conserves particle weight, i.e.
    /// 
    ///    delta W(x)  = - flux (x - 1/2) + flux (x + 1/2)
    ///
    /// where W is the weight of the particle at the point midway between the two
    /// boundaries.
    fn flux(x_i: f64, x_f: f64) -> f64 {
        if x_i.abs() < 1.0 {
            if x_f.abs() >= 1.0 {
                let v = 0.5 * (1.0 - x_i.abs()).powi(2);
                v.copysign(-x_i)
            } else if x_i * x_f >= 0.0 { // x_i and x_f have the same sign
                let v = 0.5 * (1.0 - x_f.abs()).powi(2) - 0.5 * (1.0 - x_i.abs()).powi(2);
                v.copysign(x_i - x_f)
            } else { // x_i and x_f have different signs
                let v = x_i.abs() * (1.0 - 0.5 * x_i.abs()) + x_f.abs() * (1.0 - 0.5 * x_f.abs());
                v.copysign(x_i)
            }
        }
        else if x_f.abs() < 1.0 { // and |x_i| >= 1.0
            let v = 0.5 * (1.0 - x_f.abs()).powi(2);
            v.copysign(x_f)
        } else { // both x_i and x_f outside particle
            0.0
        }
    }
}

impl Add for Cell {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            rho: self.rho + other.rho,
            j: [self.j[0] + other.j[0], self.j[1] + other.j[1], self.j[2] + other.j[2]],
            E: [self.E[0] + other.E[0], self.E[1] + other.E[1], self.E[2] + other.E[2]],
            B: [self.B[0] + other.B[0], self.B[1] + other.B[1], self.B[2] + other.B[2]],
        }
    }

}

/// Defines a equivalent MPI datatype for Cell
unsafe impl Equivalence for Cell {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        let blocklengths = [1, 1, 3, 3, 3];
        let displacements = [
            offset_of!(Cell, x) as mpi::Address,
            offset_of!(Cell, rho) as mpi::Address,
            offset_of!(Cell, j) as mpi::Address,
            offset_of!(Cell, E) as mpi::Address,
            offset_of!(Cell, B) as mpi::Address
        ];
        let mpi_double = f64::equivalent_datatype();
        let types: [&dyn Datatype; 5] = [&mpi_double; 5];
        UserDatatype::structured(5, &blocklengths, &displacements, &types)
    }
}

const GHOST_SIZE: isize = 4;
const LASER_BDY_SIZE: isize = 4;
const ABSORBING_BDY_SIZE: isize = 200;
const CONDUCTING_BDY_SIZE: isize = 4;

/// YeeGrid discretizes the electromagnetic field and associated
/// currents into a 1D Cartesian mesh of equally sized cells.
/// 
/// Maxwell's equations are solved using an explicit,
/// second-order finite-difference time-domain (FDTD) method.
/// The timestep can therefore be, at most, dx / c, where
/// dx is the grid spacing and c the speed of light.
/// 
/// Each Grid stores a ghost zone for the Grid to its left
/// and right, which must be synchronized before the
/// field advance.
pub struct YeeGrid {
    id: i32,
    ngrids: i32,
    rank_to_left: i32,
    rank_to_right: i32,
    left_bdy: Boundary,
    right_bdy: Boundary,
    left_bdy_size: isize,
    right_bdy_size: isize,
    size: usize,
    //x: f64,
    dx: f64,
    cell: Array1<Cell>,
    //laser: Box<dyn Fn(f64, f64)->f64 + 'a + Sync>,
}

impl Grid for YeeGrid {
    fn build(geometry: GridDesign) -> Self {
        let id = geometry.id;
        let numtasks = geometry.numtasks;
        let ncells = geometry.nx[id as usize];
        let offset = geometry.offset[id as usize];

        let (left_bdy, left_bdy_size) = if id == 0 && geometry.left == Boundary::Laser {
            (Boundary::Laser, LASER_BDY_SIZE)
        } else {
            (Boundary::Internal, GHOST_SIZE)
        };

        let (right_bdy, right_bdy_size) = if id == numtasks-1 {
            match geometry.right {
                Boundary::Absorbing => (Boundary::Absorbing, ABSORBING_BDY_SIZE),
                Boundary::Conducting => (Boundary::Conducting, CONDUCTING_BDY_SIZE),
                _ => (Boundary::Internal, GHOST_SIZE),
            }
        } else {
            (Boundary::Internal, GHOST_SIZE)
        };

        let empty_cell = |c| -> Cell {
            Cell {
                x: geometry.xmin + offset + ((c as f64) - (left_bdy_size as f64)) * geometry.dx,
                rho: 0.0,
                j: [0.0; 3],
                E: [0.0; 3],
                B: [0.0; 3],
            }
        };

        YeeGrid {
            id: id,
            ngrids: numtasks,
            left_bdy: left_bdy,
            right_bdy: right_bdy,
            rank_to_left: (id - 1 + numtasks) % numtasks,
            rank_to_right: (id + 1) % numtasks,
            left_bdy_size: left_bdy_size,
            right_bdy_size: right_bdy_size,
            size: ncells,
            //x: x + ((id as f64) * (ncells as f64)) * dx,
            dx: geometry.dx,
            cell: Array1::from_shape_fn(ncells + (left_bdy_size as usize) + (right_bdy_size as usize), empty_cell),
            //laser: Box::new(laser),
        }
    }

    fn rank(&self) -> i32 {
        self.id
    }

    fn ngrids(&self) -> i32 {
        self.ngrids
    }

    fn to_left(&self) -> Option<i32> {
        if self.left_bdy == Boundary::Internal {
            Some(self.rank_to_left)
        } else {
            None
        }
    }

    fn to_right(&self) -> Option<i32> {
        if self.right_bdy == Boundary::Internal {
            Some(self.rank_to_right)
        } else {
            None
        }
    }

    fn advance(&mut self, dt: f64) {
        self.advance_B(0.5 * dt);
        self.advance_E(dt);
        self.advance_B(0.5 * dt);
    }

    fn synchronize(&mut self, world: impl Communicator, laser_y: &impl Fn(f64, f64) -> f64, laser_z: &impl Fn(f64, f64) -> f64, t: f64, dt: f64) {
        // Take first and last 'left_bdy_size + right_bdy size' elements
        let send_left = self.cell.slice(s![0..2*GHOST_SIZE]).to_vec();
        let send_right = self.cell.slice(s![-2*GHOST_SIZE..]).to_vec();
        let mut recv_left: Vec<Cell> = Vec::new();
        let mut recv_right: Vec<Cell> = Vec::new();

        /* First all even-numbered grids send to the right (i.e.
           odd-numbered grids receive from the left). Then
           even-numbered grids receive from the right (and odds
           send to the left). */

        if self.rank() % 2 == 0 {
            if self.right_bdy == Boundary::Internal {
                let mut tmp = if self.rank_to_right != self.id {
                    world.process_at_rank(self.rank_to_right).synchronous_send(&send_right[..]);
                    world.process_at_rank(self.rank_to_right).receive_vec::<Cell>().0
                } else {
                    send_left.clone()
                };
                recv_right.append(&mut tmp);
            }
        } else {
            if self.left_bdy == Boundary::Internal {
                let mut tmp = if self.rank_to_left != self.id {
                    let tmp = world.process_at_rank(self.rank_to_left).receive_vec::<Cell>().0;
                    world.process_at_rank(self.rank_to_left).synchronous_send(&send_left[..]);
                    tmp
                } else {
                    send_right.clone()
                };
                recv_left.append(&mut tmp);
            }
        }

        /* And vice versa */

        if self.rank() % 2 == 0 {
            if self.left_bdy == Boundary::Internal {
                let mut tmp = if self.rank_to_left != self.id {
                    world.process_at_rank(self.rank_to_left).synchronous_send(&send_left[..]);
                    world.process_at_rank(self.rank_to_left).receive_vec::<Cell>().0
                } else {
                    send_right.clone()
                };
                recv_left.append(&mut tmp);
            }
        } else {
            if self.right_bdy == Boundary::Internal {
                let mut tmp = if self.rank_to_right != self.id {
                    let tmp = world.process_at_rank(self.rank_to_right).receive_vec::<Cell>().0;
                    world.process_at_rank(self.rank_to_right).synchronous_send(&send_right[..]);
                    tmp
                } else {
                    send_left.clone()
                };
                recv_right.append(&mut tmp);
            }
        }

        /* Now update all the ghost zones */

        if self.left_bdy == Boundary::Internal {
            assert_eq!( recv_left.len(), 2 * (GHOST_SIZE as usize));
        }

        if self.right_bdy == Boundary::Internal {
            assert_eq!( recv_right.len(), 2 * (GHOST_SIZE as usize));
        }

        let recv_left = Array1::from(recv_left);
        let recv_right = Array1::from(recv_right);

        if self.left_bdy == Boundary::Internal {
            self.cell
                .slice_mut(s![0..GHOST_SIZE])
                .zip_mut_with(
                    & recv_left.slice(s![0..GHOST_SIZE]),
                    Cell::overlay_ghost
                );
            self.cell
                .slice_mut(s![GHOST_SIZE..2*GHOST_SIZE])
                .zip_mut_with(
                    & recv_left.slice(s![GHOST_SIZE..]),
                    Cell::overlay
                );
        }

        if self.right_bdy == Boundary::Internal {
            self.cell
                .slice_mut(s![-2*GHOST_SIZE..-GHOST_SIZE])
                .zip_mut_with(
                    & recv_right.slice(s![0..GHOST_SIZE]),
                    Cell::overlay
                );
            self.cell
                .slice_mut(s![-GHOST_SIZE..])
                .zip_mut_with(
                    & recv_right.slice(s![GHOST_SIZE..]),
                    Cell::overlay_ghost
                );
        }

        /* Load boundary conditions */

        if self.left_bdy == Boundary::Laser {
            let start = (self.left_bdy_size - 2) as usize;
            let x = self.cell[[start]].x;
            let r = SPEED_OF_LIGHT * dt / self.dx;
            self.cell[[start]].E[1] += 2.0 * r * laser_y(t, x); // just these two works...
            self.cell[[start]].E[2] += 2.0 * r * laser_z(t, x);
        }

        if self.right_bdy == Boundary::Absorbing {
            let (xmin, xmax) = (self.cell[[self.size + (self.left_bdy_size as usize)]].x, self.cell[[self.size + (self.left_bdy_size as usize) + (self.right_bdy_size as usize) - 1]].x);
            let sigma_max = 10.0 / (self.right_bdy_size as f64);
            self.cell.slice_mut(s![-self.right_bdy_size+1..]).map_inplace(|c| {
                let sigma = sigma_max * (c.x - xmin) / (xmax - xmin);
                c.E[0] = (1.0 - sigma) * c.E[0];
                c.E[1] = (1.0 - sigma) * c.E[1];
                c.E[2] = (1.0 - sigma) * c.E[2];
                c.B[0] = (1.0 - sigma) * c.B[0];
                c.B[1] = (1.0 - sigma) * c.B[1];
                c.B[2] = (1.0 - sigma) * c.B[2];
            });
            self.cell.slice_mut(s![-2..]).map_inplace(|c| {
                c.E = [0.0; 3];
                c.B = [0.0; 3];
            });
        } else if self.right_bdy == Boundary::Conducting {
            let start: usize = self.cell.len() - (self.right_bdy_size as usize);
            // handle first cell in ghost size, which is bisected by surface
            self.cell[[start]].E[0] = 0.0;
            self.cell[[start]].B[1] = 0.0;
            self.cell[[start]].B[2] = 0.0;

            for i in 1..(self.right_bdy_size as usize) {
                self.cell[[start + i]].E[0] = -self.cell[[start - i]].E[0]; // clamp to zero at bdy
                self.cell[[start + i]].E[1] = self.cell[[start + 1 - i]].E[1]; // zero_grad
                self.cell[[start + i]].E[2] = self.cell[[start + 1 - i]].E[2]; // zero_grad
                self.cell[[start + i]].B[0] = self.cell[[start + 1 - i]].B[0]; // zero_grad
                self.cell[[start + i]].B[1] = -self.cell[[start - i]].B[1]; // clamp to zero at bdy
                self.cell[[start + i]].B[2] = -self.cell[[start - i]].B[2]; // clamp to zero at bdy
            }
        }
    }

    #[allow(non_snake_case)]
    fn fields_at(&self, c: isize, xi: f64) -> ([f64; 3], [f64; 3]) {
        let j = (c + self.left_bdy_size) as usize;
        assert!(j >= self.left_bdy_size as usize);
        assert!(j < self.cell.len() - (self.right_bdy_size as usize));

        let E = [
            self.cell[j-1].E[0] * Cell::weight(0.5 + xi)
            + self.cell[j].E[0] * Cell::weight(0.5 - xi)
            + self.cell[j+1].E[0] * Cell::weight(1.5 - xi),
            self.cell[j-1].E[1] * Cell::weight(1.0 + xi)
            + self.cell[j].E[1] * Cell::weight(xi)
            + self.cell[j+1].E[1] * Cell::weight(1.0 - xi)
            + self.cell[j+2].E[1] * Cell::weight(2.0 - xi),
            self.cell[j-1].E[2] * Cell::weight(1.0 + xi)
            + self.cell[j].E[2] * Cell::weight(xi)
            + self.cell[j+1].E[2] * Cell::weight(1.0 - xi)
            + self.cell[j+2].E[2] * Cell::weight(2.0 - xi)
        ];

        let B = [
            self.cell[j].B[0],
            self.cell[j-1].B[1] * Cell::weight(0.5 + xi)
            + self.cell[j].B[1] * Cell::weight(0.5 - xi)
            + self.cell[j+1].B[1] * Cell::weight(1.5 - xi),
            self.cell[j-1].B[2] * Cell::weight(0.5 + xi)
            + self.cell[j].B[2] * Cell::weight(0.5 - xi)
            + self.cell[j+1].B[2] * Cell::weight(1.5 - xi)
        ];

        (E, B)
    }

    fn xmin(&self) -> f64 {
        self.cell[self.left_bdy_size as usize].x
    }

    fn dx(&self) -> f64 {
        self.dx
    }

    fn size(&self) -> usize {
        self.size
    }

    fn clear(&mut self) {
        //use ndarray::parallel::prelude::*;
        self.cell.par_map_inplace(|c| {
            c.rho = 0.0;
            c.j = [0.0; 3];
        });
    }

    fn deposit<P: Particle + Send + Sync>(&mut self, pt: &[P], dt: f64) {
        let dx = self.dx;
        let size = self.cell.len();
        let left_bdy_size = self.left_bdy_size;
        let mut j: Vec<Vec<[f64; 4]>> = Vec::new();

        if pt.is_empty() {
            return;
        }

        // divide particles into nthread chunks
        let nthreads = rayon::current_num_threads();
        // chunk length cannot be zero
        let chunk_len = if pt.len() > nthreads {
            pt.len() / nthreads
        } else {
            pt.len() // which is > 0
        };

        pt.par_chunks(chunk_len)
            .map(|chunk: &[P]| -> Vec<[f64; 4]> {
                let mut j = vec![[0.0; 4]; size];
                chunk.iter().for_each(|pt| {
                    let (c, x, prev_x) = pt.location();
                    let macrocharge = pt.weight() * pt.charge();
                    let velocity = pt.velocity();
                    let index = (c + left_bdy_size) as usize;
                    assert!(index >= 2);
                    assert!(index < j.len() - 2);

                    /* current density = charge * amount of particle / (area * dt)
                       [Cells have unit cross-sectional area in x-direction.] */
    
                    j[index  ][0] += macrocharge * Cell::flux( 0.5 - prev_x,  0.5 - x) / dt;
                    j[index-1][0] += macrocharge * Cell::flux(-0.5 - prev_x, -0.5 - x) / dt;
                    j[index-2][0] += macrocharge * Cell::flux(-1.5 - prev_x, -1.5 - x) / dt;
                    j[index+1][0] += macrocharge * Cell::flux( 1.5 - prev_x,  1.5 - x) / dt;
                    j[index+2][0] += macrocharge * Cell::flux( 2.5 - prev_x,  2.5 - x) / dt;
    
                    /* In y- and z-directions, just weight the perpendicular current
                       density j_perp:
                       j_perp = charge * velocity / (area * dx) */
    
                    j[index-1][1] += macrocharge * velocity[1] * Cell::weight(1.0 + x) / dx;
                    j[index  ][1] += macrocharge * velocity[1] * Cell::weight(      x) / dx;
                    j[index+1][1] += macrocharge * velocity[1] * Cell::weight(1.0 - x) / dx;
                    j[index+2][1] += macrocharge * velocity[1] * Cell::weight(2.0 + x) / dx;
    
                    j[index-1][2] += macrocharge * velocity[2] * Cell::weight(1.0 + x) / dx;
                    j[index  ][2] += macrocharge * velocity[2] * Cell::weight(      x) / dx;
                    j[index+1][2] += macrocharge * velocity[2] * Cell::weight(1.0 - x) / dx;
                    j[index+2][2] += macrocharge * velocity[2] * Cell::weight(2.0 + x) / dx;

                    /* Charge density = macrocharge / (cell_area * dx) */

                    j[index-1][3] += macrocharge * Cell::weight(1.0 + x) / dx;
                    j[index  ][3] += macrocharge * Cell::weight(      x) / dx;
                    j[index+1][3] += macrocharge * Cell::weight(1.0 - x) / dx;
                    j[index-2][3] += macrocharge * Cell::weight(2.0 - x) / dx;
                });
                j
            })
            .collect_into_vec(&mut j);

        // Add up all the currents
        let mut total: Vec<[f64; 4]> = vec![[0.0; 4]; size];

        // this isn't necessarily true due to chunking
        //assert!( j.len() == nthreads );

        for i in 0..size {
            for k in 0..j.len() { // not 0..nthreads
                total[i][0] += j[k][i][0];
                total[i][1] += j[k][i][1];
                total[i][2] += j[k][i][2];
                total[i][3] += j[k][i][3];
            }
        }
        /*
        for i in &total {
            println!("{:?}", i);
        }
        */
        // Add to cells
        for (c, j) in self.cell.iter_mut().zip(total) {
            c.j[0] += j[0];
            c.j[1] += j[1];
            c.j[2] += j[2];
            c.rho  += j[3];
        }
    }
 
    #[allow(non_snake_case)]
    fn initialize(&mut self, world: impl Communicator) {
        use mpi::collective::SystemOperation;
        // Assume that the initial charges and currents have been deposited
        // and synchronized

        // Determine the total charge and current on the grid
        let (local_rho, local_j) = self.cell
            .slice(s![self.left_bdy_size..-self.right_bdy_size])
            .fold( (0.0, [0.0; 3]), |rt, c| {
                (rt.0 + c.rho, [rt.1[0] + c.j[0], rt.1[1] + c.j[1], rt.1[2] + c.j[2]])
            });

        if self.rank() == 0 {
            let mut domain_rho = 0.0;
            let mut domain_j = [0.0; 3];

            // Root needs the total charge and current across all grids
            world
                .process_at_rank(0)
                .reduce_into_root(&local_rho, &mut domain_rho, SystemOperation::sum());
            world
                .process_at_rank(0)
                .reduce_into_root(&local_j[..], &mut domain_j[..], SystemOperation::sum());

            //println!("Total charge = {:e}, current = {:?}", domain_rho * self.dx, domain_j * self.dx);

            /*
            Loop over the entire domain setting the initial fields, solving

            Gauss's law for magnetism: dBx/dx = 0
            Gauss's law:               dEx/dx = rho/e_0
            Ampere's law:              curl B = (0, -dBz/dx, dBy/dx)
                                       = mu_0 (jx + e_0 dEx/dt, jy, jz)

            This means Bx, Ey and Ez are initially everywhere zero.

            Boundary conditions for Ex are set by recalling that, for a 
            a sheet of charge of infinite extent in the y- and z- directions,
            the resultant field is |Ex| = sigma / 2 e0 where sigma = charge
            per unit area.

            For an infinite current sheet with current density (0, jy, jz)
            and thickness L and located at x = x0, the magnetic field at a
            position x < x0 is given by B = 1/2 mu0 L (0, -jz, jy).
            */
            
            let domain_E = [
                -domain_rho * self.dx / (2.0 * VACUUM_PERMITTIVITY),
                0.0,
                0.0
            ];
            let domain_B = [
                0.0,
                -VACUUM_PERMEABILITY * domain_j[2] * self.dx / 2.0,
                VACUUM_PERMEABILITY * domain_j[1] * self.dx / 2.0
            ];

            // The boundary conditions have already been loaded, so
            // superpose the domain fields on the laser

            self.cell.slice_mut(s![0..self.left_bdy_size]).map_inplace(|c| {
                c.E[0] += domain_E[0];
                c.E[1] += domain_E[1];
                c.E[2] += domain_E[2];
                c.B[0] += domain_B[0];
                c.B[1] += domain_B[1];
                c.B[2] += domain_B[2];
            });
        } else {
            // Send our charges and currents to root
            world
                .process_at_rank(0)
                .reduce_into(&local_rho, SystemOperation::sum());
            world
                .process_at_rank(0)
                .reduce_into(&local_j[..], SystemOperation::sum());
            
            // Now, stand by to receive fields in the left ghost zone
        }

        if let Some(rank) = self.to_left() {
            let tmp = world.process_at_rank(rank).receive_vec::<Cell>().0;
            // I only want the fields from tmp, to fill the left ghost zone
            for (a, b) in self.cell.slice_mut(s![..self.left_bdy_size]).iter_mut().zip(&tmp[..]) {
                a.E = b.E;
                a.B = b.B;
            }
        }

        // Loop over local cells [left_bdy_size..]
        let start: usize = self.left_bdy_size as usize;
        let end: usize = self.cell.len();
        for i in start..end {
            self.cell[[i]].E[0] = self.cell[[i-1]].E[0] + self.dx * self.cell[[i]].rho / VACUUM_PERMITTIVITY;
            self.cell[[i]].B[1] = self.cell[[i-1]].B[1] + VACUUM_PERMEABILITY * self.dx * self.cell[[i]].j[2];
            self.cell[[i]].B[2] = self.cell[[i-1]].B[2] - VACUUM_PERMEABILITY * self.dx * self.cell[[i]].j[1];
        }

        if let Some(rank) = self.to_right() {
            // send ghost zone to rightlet
            let send_right = self.cell.slice(s![-2*GHOST_SIZE..]).to_vec();
            world.process_at_rank(rank).synchronous_send(&send_right[..]);
        } // else hit the absorbing boundary
    }

    fn write_data(&self, world: impl Communicator, dir: &str, index: usize) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let id = self.id;

        if id == 0 {
            let mut global = self.interpolate();
            assert_eq!( global.len(), self.size );
            for recv_rank in 1..self.ngrids() {
                let (recv, _) = world.process_at_rank(recv_rank).receive_vec::<Cell>();
                global.extend(recv);
            }

            let filename = format!("{}/{}_grid.dat", dir, index);
            let mut file = File::create(filename)?;

            for cell in global.iter() {
                writeln! (
                    file,
                    "{:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {}",
                    cell.x, cell.rho,
                    cell.j[0], cell.j[1], cell.j[2],
                    cell.E[0], cell.E[1], cell.E[2],
                    cell.B[0], cell.B[1], cell.B[2]
                )?;
            }
        } else {
            let send = self.interpolate();
            world.process_at_rank(0).synchronous_send(&send[..]);
        }

        Ok(())
    }

    fn min_size() -> usize {
        (2 * GHOST_SIZE) as usize
    }

    fn em_field_energy(&self, world: &impl Communicator) -> f64 {
        use mpi::collective::SystemOperation;

        let (sum_e2, sum_b2) = self.cell
            .slice(s![self.left_bdy_size..-self.right_bdy_size])
            .fold((0.0, 0.0), |(e2, b2), c|
                (
                    e2 + c.E[0].powi(2) + c.E[1].powi(2) + c.E[2].powi(2),
                    b2 + c.B[0].powi(2) + c.B[1].powi(2) + c.B[2].powi(2),
                )
            );

        let local = 0.5 * (VACUUM_PERMITTIVITY * sum_e2 + sum_b2 / VACUUM_PERMEABILITY) * self.dx;

        if self.id == 0 {
            let mut global = 0.0;
            world.process_at_rank(0).reduce_into_root(&local, &mut global, SystemOperation::sum());
            global
        } else {
            world.process_at_rank(0).reduce_into(&local, SystemOperation::sum());
            local
        }
    }
}

impl YeeGrid {
    /// Interpolate all grid quantities to the cell left boundary,
    /// having stripped away ghost cells.
    fn interpolate(&self) -> Vec<Cell> {
        let end = -self.right_bdy_size;//if self.right_bdy == Boundary::Absorbing {-1} else {-self.right_bdy_size};
        let v: Vec<Cell> = self.cell
            .slice(s![self.left_bdy_size-1..end])
            .to_vec();

        let intrp: Vec<Cell> = v.windows(2)
            .map(|c| -> Cell {
                let (a, b) = (c[0], c[1]);
                Cell {
                    x: b.x,
                    rho: b.rho,
                    j: [0.5 * (a.j[0] + b.j[0]), b.j[1], b.j[2]],
                    E: [0.5 * (a.E[0] + b.E[0]), b.E[1], b.E[2]],
                    B: [b.B[0], 0.5 * (a.B[1] + b.B[1]), 0.5 * (a.B[2] + b.B[2])],
                }
            })
            .collect();
        
        intrp
    }

    /// Advance the magnetic fields components over an interval `dt`.
    #[allow(non_snake_case)]
    fn advance_B(&mut self, dt: f64) {
        let start: usize = 0;
        let end: usize = self.cell.len() - 1;
        //println!("looping over indices {:?}", range);
        for i in start..end {
            // nothing happens to Bx in 1D
            self.cell[[i]].B[1] += dt * (self.cell[[i+1]].E[2] - self.cell[[i]].E[2]) / self.dx;
            self.cell[[i]].B[2] += dt * (self.cell[[i]].E[1] - self.cell[[i+1]].E[1]) / self.dx;
        }
    }

    /// Advance the electric fields components over an interval `dt`.
    #[allow(non_snake_case)]
    fn advance_E(&mut self, dt: f64) {
        // Handle first cell specially, with Silver-Muller bc
        let kappa = 2.0 * SPEED_OF_LIGHT * dt / (SPEED_OF_LIGHT * dt + self.dx);
        let sigma = 1.0 - kappa;
        self.cell[[0]].E[0] = 0.0;
        self.cell[[0]].E[1] = sigma * self.cell[[0]].E[1] - SPEED_OF_LIGHT * kappa * self.cell[[0]].B[2];
        self.cell[[0]].E[2] = sigma * self.cell[[0]].E[2] + SPEED_OF_LIGHT * kappa * self.cell[[0]].B[1];
        // and then everything else
        let start: usize = 1;
        let end: usize = self.cell.len(); // all the way to the end
        for i in start..end {
            self.cell[[i]].E[0] += -dt * self.cell[[i]].j[0] / VACUUM_PERMITTIVITY;
            self.cell[[i]].E[1] += dt * SPEED_OF_LIGHT_SQD * (self.cell[[i-1]].B[2] - self.cell[[i]].B[2]) / self.dx - dt * self.cell[[i]].j[1] / VACUUM_PERMITTIVITY;
            self.cell[[i]].E[2] += dt * SPEED_OF_LIGHT_SQD * (self.cell[[i]].B[1] - self.cell[[i-1]].B[1]) / self.dx - dt * self.cell[[i]].j[2] / VACUUM_PERMITTIVITY;
        }
    }

    /// Bilinear filter of charge and current density,
    /// suppresses signal at the Nyquist frequency.
    #[allow(unused)]
    fn smooth_currents(&mut self, alpha: f64) {
        let mut smoothed = self.cell.clone();
        let windows = self.cell.windows(3);

        smoothed
            .slice_mut(s![1..-1])
            .iter_mut()
            .zip(windows)
            .for_each(|(c, w)| {
                // j_i = alpha j_i + (1 - alpha) (j_{i-1} + j_{i+1}) / 2
                c.j[0] = alpha * w[[1]].j[0] + 0.5 * (1.0 - alpha) * (w[[0]].j[0] + w[[2]].j[0]);
                c.j[1] = alpha * w[[1]].j[1] + 0.5 * (1.0 - alpha) * (w[[0]].j[1] + w[[2]].j[1]);
                c.j[2] = alpha * w[[1]].j[2] + 0.5 * (1.0 - alpha) * (w[[0]].j[2] + w[[2]].j[2]);
            });

        std::mem::swap(&mut self.cell, &mut smoothed);
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts;
    use std::sync::Once;
    use super::*;

    // This nonsense is necessary because MPI_INIT can only be
    // called once, and we have two tests that require it.
    static INIT: Once = Once::new();
    static mut UNIVERSE: Option<mpi::environment::Universe> = None;

    #[test]
    fn laser_bc() {
        let universe = unsafe {
            INIT.call_once(|| {
                UNIVERSE = mpi::initialize();
            });
            UNIVERSE.as_ref().unwrap()
        };
        //let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let xmin = -10.0e-6;
        let dx = 1.0e-6 / 100.0;
        let mut t = -15.0e-6 / SPEED_OF_LIGHT;
        let dt = 0.95 * dx / SPEED_OF_LIGHT;
        let nsteps: usize = (t.abs() / dt) as usize; // end at t = 0

        let omega = 2.0 * consts::PI * SPEED_OF_LIGHT / 1.0e-6;
        let emax = 1.0e10;
        let laser_y = |t: f64, z: f64| {
            let phi = omega * (t - z / SPEED_OF_LIGHT);
            if phi.abs() < 8.0 * consts::PI {
                emax * phi.sin() * (phi/16.0).cos().powi(2)
            } else {
                0.0
            }
        };
        let laser_z = |_t, _z| {0.0};

        let design = GridDesign::unbalanced(world, 2000, xmin, dx, YeeGrid::min_size(), Boundary::Laser, Boundary::Absorbing);
        let mut grid = YeeGrid::build(design);

        for _i in 0..=nsteps {
            grid.synchronize(world, &laser_y, &laser_z, t, dt);
            grid.advance(dt);
            t += dt;
        }

        let (e, _) = grid.fields_at(1000 + 25, 0.0); // x = lambda/4
        println!("x = lambda/4: ey/e0 = {:.6e}, expected = {:.6e}", e[1]/emax, -(consts::PI/32.0).cos().powi(2));

        let (e, _) = grid.fields_at(1000 + 125, 0.0); // x = 5 lambda/4
        println!("x = 5 lambda/4: ey/e0 = {:.6e}, expected = {:.6e}", e[1]/emax, -(5.0*consts::PI/32.0).cos().powi(2));

        let em_energy = grid.em_field_energy(&world);
        // total energy = epsilon_0 A \int |E|^2 dx
        // E = E_0 sin(k x) cos(k x/16) at t = 0 for |k x| < 8 pi
        let target = VACUUM_PERMITTIVITY * emax.powi(2) * (3.0 * consts::PI * SPEED_OF_LIGHT / omega);

        println!("t = {:.3e}, em_energy = {:.6e}, expected = {:.6e}", t, em_energy, target);
        assert!( (em_energy - target).abs() / target < 1.0e-3);
    }

    #[test]
    fn conducting_bc() {
        let universe = unsafe {
            INIT.call_once(|| {
                UNIVERSE = mpi::initialize();
            });
            UNIVERSE.as_ref().unwrap()
        };
        //let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let xmin = -10.0e-6;
        let dx = 1.0e-6 / 100.0;
        let mut t = -15.0e-6 / SPEED_OF_LIGHT;
        let dt = 0.95 * dx / SPEED_OF_LIGHT;
        let nsteps: usize = (35.0e-6 / (SPEED_OF_LIGHT * dt)) as usize; // 35, end after laser has bounced

        let omega = 2.0 * consts::PI * SPEED_OF_LIGHT / 1.0e-6;
        let emax = 1.0e10;

        let laser_y = |t: f64, z: f64| {
            let phi = omega * (t - z / SPEED_OF_LIGHT);
            if phi.abs() < 8.0 * consts::PI {
                -emax * (phi.sin() + (2.0/16.0) * phi.cos() * (phi/16.0).tan()) * (phi/16.0).cos().powi(2)
            } else {
                0.0
            }
        };

        let laser_z = |t: f64, z: f64| {
            let phi = omega * (t - z / SPEED_OF_LIGHT);
            if phi.abs() < 8.0 * consts::PI {
                emax * (phi.cos() + (2.0/16.0) * phi.sin() * (phi/16.0).tan()) * (phi/16.0).cos().powi(2)
            } else {
                0.0
            }
        };

        let design = GridDesign::unbalanced(world, 100 * 20, xmin, dx, YeeGrid::min_size(), Boundary::Laser, Boundary::Conducting);
        let mut grid = YeeGrid::build(design);

        for _i in 0..=nsteps {
            grid.synchronize(world, &laser_y, &laser_z, t, dt);
            grid.advance(dt);
            t += dt;
        }

        //grid.write_data(world, "output", 0).unwrap();

        let em_energy = grid.em_field_energy(&world);
        // total energy = epsilon_0 A \int |E|^2 dx
        let target = VACUUM_PERMITTIVITY * emax.powi(2) * (193.0 / 32.0) * (consts::PI * SPEED_OF_LIGHT / omega);
        let error = (em_energy - target).abs() / target;

        println!("t = {:.3e}, em_energy = {:.6e}, expected = {:.6e}, error = {:.3e}", t, em_energy, target, error);
        assert!(error < 1.0e-3);
    }

    #[test]
    fn periodic_bc() {
        let universe = unsafe {
            INIT.call_once(|| {
                UNIVERSE = mpi::initialize();
            });
            UNIVERSE.as_ref().unwrap()
        };

        let world = universe.world();

        let xmin = -10.0e-6;
        let dx = 1.0e-6 / 100.0;
        let mut t = -5.0e-6 / SPEED_OF_LIGHT;
        let dt = 0.95 * dx / SPEED_OF_LIGHT;
        let nsteps: usize = (10.0e-6 / (SPEED_OF_LIGHT * dt)) as usize;

        let omega = 2.0 * consts::PI * SPEED_OF_LIGHT / 1.0e-6;
        let jmax = omega * VACUUM_PERMITTIVITY * 1.0e5; // max e should be 1.0e5
        let no_laser = |_t, _x| {0.0};

        let design = GridDesign::unbalanced(world, 100 * 20, xmin, dx, YeeGrid::min_size(), Boundary::Internal, Boundary::Internal);
        let mut grid = YeeGrid::build(design);

        println!("Initialized grid with {} cells on {} tasks, t = {:.2e}", grid.size(), grid.ngrids(), t);
        println!("Grid to left = {:?}, to right = {:?}", grid.to_left(), grid.to_right());

        for _i in 0..nsteps {
            grid.cell[[50 * 20]].j[2] = {
                let phi = omega * t;
                if phi.abs() < 4.0 * consts::PI {
                    jmax * (phi.sin() + (2.0/8.0) * phi.cos() * (phi/8.0).tan()) * (phi/8.0).cos().powi(2)
                } else {
                    0.0
                }
            };

            grid.synchronize(world, &no_laser, &no_laser, t, dt);
            grid.advance(dt);
            t += dt;
        }
        
        //grid.write_data(world, "output", 0).unwrap();
        let initial_em_energy = grid.em_field_energy(&world);
        println!("Updated grid to t = {:.2e}, em_field_energy = {:.6e}", t, initial_em_energy);

        for _i in 0..nsteps {
            grid.cell[[50 * 20]].j[2] = {
                let phi = omega * t;
                if phi.abs() < 4.0 * consts::PI {
                    jmax * (phi.sin() + (2.0/8.0) * phi.cos() * (phi/8.0).tan()) * (phi/8.0).cos().powi(2)
                } else {
                    0.0
                }
            };

            grid.synchronize(world, &no_laser, &no_laser, t, dt);
            grid.advance(dt);
            t += dt;
        }

        //grid.write_data(world, "output", 1).unwrap();
        let final_em_energy = grid.em_field_energy(&world);
        println!("Updated grid to t = {:.2e}, em_field_energy = {:.6e}", t, final_em_energy);

        assert!( (initial_em_energy - final_em_energy).abs() / initial_em_energy < 1.0e-6 );
    }
}
