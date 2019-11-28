use std::fmt;
use std::f64;
use std::cmp::Ordering;
use num::complex::Complex;

use mpi::traits::*;
use mpi::datatype::UserDatatype;
use memoffset::*;

use crate::constants::*;
use crate::particle::*;
use crate::particle::vec3::*;

#[derive(Copy,Clone)]
pub struct Photon {
    cell: isize,
    prev_x: f64,
    x: f64,
    weight: f64,
    k: Vec3, // momentum / mc
    pol: [Complex<f64>; 2],
    basis: [Vec3; 2],
    chi: f64,
    tau: [f64; 2],
}

impl fmt::Debug for Photon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[photon: x = {} + {}, k = {:?}]", self.cell, self.x, self.k)
    }
}

unsafe impl Equivalence for Photon {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        let blocklengths = [1, 1, 1, 1, 1, 2, 4, 1, 2];
        let displacements = [
            offset_of!(Photon, cell) as mpi::Address,
            offset_of!(Photon, prev_x) as mpi::Address,
            offset_of!(Photon, x) as mpi::Address,
            offset_of!(Photon, weight) as mpi::Address,
            offset_of!(Photon, k) as mpi::Address,
            offset_of!(Photon, pol) as mpi::Address,
            offset_of!(Photon, basis) as mpi::Address,
            offset_of!(Photon, chi) as mpi::Address,
            offset_of!(Photon, tau) as mpi::Address,
        ];
        let types: [&dyn Datatype; 9] = [
            &isize::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &Vec3::equivalent_datatype(),
            &f64::equivalent_datatype(), // fingers crossed
            &Vec3::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
        ];
        UserDatatype::structured(9, &blocklengths, &displacements, &types)
    }
}

impl PartialOrd for Photon {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cell.cmp(&other.cell))
    }
}

impl PartialEq for Photon {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}

impl Particle for Photon {
    fn create(cell: isize, x: f64, u: &[f64; 3], weight: f64, dx: f64, dt: f64) -> Self {
        let k = Vec3::new_from_slice(u);
        let k0 = k.norm_sqr().sqrt();
        let prev_x = x - SPEED_OF_LIGHT * k.x * dt / (k0 * dx);
        Photon {
            cell: cell,
            prev_x: prev_x,
            x: x,
            weight: weight,
            k: k,
            pol: [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
            basis: [k, k], // temporary
            chi: 0.0,
            tau: [f64::INFINITY; 2],
        }
    }

    fn with_optical_depth(&self, tau: f64) -> Self {
        let mut pt = *self;
        pt.tau = [tau; 2];
        pt
    }

    fn spin_state(&self) -> Option<f64> {
        Some(self.helicity())
    }

    fn spin_state_name(&self) -> Option<&'static str> {
        Some("helicity")
    }

    #[allow(non_snake_case)]
    fn push(&mut self, E: &[f64; 3], B: &[f64; 3], dx: f64, dt: f64) {
        let E = Vec3::new_from_slice(E);
        let B = Vec3::new_from_slice(B);
        
        // velocity in SI units
        let k0 = self.k.norm_sqr().sqrt();
        let v = SPEED_OF_LIGHT * self.k / k0;

        // quantum parameter
        self.chi = ((k0 * E + SPEED_OF_LIGHT * self.k.cross(B)).norm_sqr() - (E * self.k).powi(2)).sqrt() / CRITICAL_FIELD;

        // then the position
        self.prev_x = self.x;
        let dxi = v.x * dt / dx;
        assert!(dxi < 1.0);
        self.x = self.x + dxi;

        // adjust for crossing a cell boundary
        let floor = self.x.floor();
        self.cell = if floor < 0.0 {
            self.cell - 1
        } else if floor > 0.0 {
            self.cell + 1
        } else {
            self.cell
        };
        //self.cell += floor as usize;
        self.prev_x -= floor;
        self.x -= floor;
    }

    fn location(&self) -> (isize, f64, f64) {
        (self.cell, self.x, self.prev_x)
    }

    fn shift_cell(&mut self, delta: isize) {
        self.cell = self.cell + delta;
    }

    fn charge(&self) -> f64 {
        0.0
    }

    fn mass(&self) -> f64 {
        0.0
    }

    fn work(&self) -> f64 {
        0.0
    }

    fn momentum(&self) -> [f64; 3] {
        let p = self.k  * ELECTRON_MASS_MEV;
        [p.x, p.y, p.z]
    }

    fn velocity(&self) -> [f64; 3] {
        let k0 = self.k.norm_sqr().sqrt();
        let v = SPEED_OF_LIGHT * self.k / k0;
        [v.x, v.y, v.z]
    }

    fn energy(&self) -> f64 {
        self.k.norm_sqr().sqrt() * ELECTRON_MASS_MEV
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn chi(&self) -> f64 {
        self.chi
    }
}

#[allow(unused)]
impl Photon {
    pub fn with_polarization_along(&self, dir: [f64; 3]) -> Self {
        let mut pt = self.clone();
        // k, basis[0] and basis[1] form a right-handed triad
        pt.basis[0] = Vec3::new(dir[0], dir[1], dir[2]).normalize();
        pt.basis[1] = pt.k.cross(pt.basis[0]).normalize();
        // photon polarized along dir, initial phase set to zero
        pt.pol[0] = Complex::new(1.0, 0.0);
        pt.pol[1] = Complex::new(0.0, 0.0);
        pt
    }

    pub fn linear_polarization_along(&self, dir: [f64; 3]) -> f64 {
        let dir = Vec3::new_from_slice(&dir).normalize();
        let amplitude = self.pol[0] * (dir * self.basis[0]) + self.pol[1] * (dir * self.basis[1]);
        amplitude.norm_sqr()
    }

    // defined with respect to its propagation direction k
    // recall e_pm = (e_1 \pm i e_2) / sqrt(2)
    // so a_+ = (a_1 - i a_2) / sqrt(2)
    pub fn helicity(&self) -> f64 {
        let amplitude = (self.pol[0] - Complex::new(0.0, 1.0) * self.pol[1]) / 2.0f64.sqrt();
        amplitude.norm_sqr()
    }
}