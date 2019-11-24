use std::fmt;
use std::cmp::Ordering;

use mpi::traits::*;
use mpi::datatype::UserDatatype;
use memoffset::*;

use crate::constants::*;
use crate::particle::*;
use crate::particle::vec3::*;

#[derive(Copy,Clone)]
#[allow(non_snake_case)]
pub struct Ion {
    cell: isize,
    prev_x: f64,
    x: f64,
    weight: f64,
    Z_star: f64,
    Z: f64,
    A: f64,
    gamma_m1: f64,
    u: Vec3,
    work: f64,
}

impl fmt::Debug for Ion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[ion: Z* = {}, A = {}, x = {} + {}, p/(Mc) = {:?}]", self.Z_star, self.A, self.cell, self.x, self.u)
    }
}

unsafe impl Equivalence for Ion {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        let blocklengths = [1; 10];
        let displacements = [
            offset_of!(Ion, cell) as mpi::Address,
            offset_of!(Ion, prev_x) as mpi::Address,
            offset_of!(Ion, x) as mpi::Address,
            offset_of!(Ion, weight) as mpi::Address,
            offset_of!(Ion, Z_star) as mpi::Address,
            offset_of!(Ion, Z) as mpi::Address,
            offset_of!(Ion, A) as mpi::Address,
            offset_of!(Ion, gamma_m1) as mpi::Address,
            offset_of!(Ion, u) as mpi::Address,
            offset_of!(Ion, work) as mpi::Address,
        ];
        let types: [&dyn Datatype; 10] = [
            &isize::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &Vec3::equivalent_datatype(),
            &f64::equivalent_datatype(),
        ];
        UserDatatype::structured(10, &blocklengths, &displacements, &types)
    }
}

impl PartialOrd for Ion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cell.cmp(&other.cell))
    }
}

impl PartialEq for Ion {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}

impl Particle for Ion {
    fn create(cell: isize, x: f64, u: &[f64; 3], weight: f64, dx: f64, dt: f64) -> Self {
        let u = Vec3::new_from_slice(u);
        // gamma = sqrt(1 + u^2)
        // gamma - 1 = u^2 / (1 + gamma)
        let gamma_m1 = u.norm_sqr() / (1.0 + (1.0 + u.norm_sqr()).sqrt());
        let v = SPEED_OF_LIGHT * u / (1.0 + gamma_m1);
        let prev_x = x - v.x * dt / dx;
        Ion {
            cell: cell,
            prev_x: prev_x,
            x: x,
            weight: weight,
            Z_star: 1.0,
            Z: 1.0,
            A: 1.0,
            gamma_m1: gamma_m1,
            u: u,
            work: 0.0,
        }
    }

    fn location(&self) -> (isize, f64, f64) {
        (self.cell, self.x, self.prev_x)
    }

    fn shift_cell(&mut self, delta: isize) {
        self.cell += delta;
    }

    fn charge(&self) -> f64 {
        self.Z_star * ELEMENTARY_CHARGE
    }

    fn mass(&self) -> f64 {
        self.A * PROTON_MASS
    }

    fn velocity(&self) -> [f64; 3] {
        let v = SPEED_OF_LIGHT * self.u / (1.0 + self.gamma_m1);
        [v.x, v.y, v.z]
    }

    fn energy(&self) -> f64 {
        self.gamma_m1 * (self.A * PROTON_MASS / ELECTRON_MASS) * ELECTRON_MASS_MEV
    }

    fn work(&self) -> f64 {
        self.work
    }

    fn momentum(&self) -> [f64; 3] {
        let p = self.u * (self.A * PROTON_MASS / ELECTRON_MASS) * ELECTRON_MASS_MEV;
        [p.x, p.y, p.z]
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn chi(&self) -> f64 {
        0.0
    }

    fn with_optical_depth(&self, _tau: f64) -> Self {
        *self
    }

    #[allow(non_snake_case)]
    fn push(&mut self, E: &[f64; 3], B: &[f64; 3], dx: f64, dt: f64) {
        let E = Vec3::new_from_slice(E);
        let B = Vec3::new_from_slice(B);
        
        // velocity in SI units
        let v = SPEED_OF_LIGHT * self.u / (1.0 + self.gamma_m1);

        // u_i = u_{i-1/2} + (q dt/2 m c) (E + v_{i-1/2} x B)
        let q = self.Z_star * ELEMENTARY_CHARGE;
        let M = self.A * PROTON_MASS;

        let alpha = q * dt / (2.0 * M * SPEED_OF_LIGHT);
        let u_half = self.u + alpha * (E + v.cross(B));
        self.work += q * SPEED_OF_LIGHT * (u_half * E) * dt / (1.0 + u_half * u_half).sqrt();

        // u' =  u_{i-1/2} + (q dt/2 m c) (2 E + v_{i-1/2} x B)
        let u_prime = u_half + alpha * E;
        let gamma_prime_sqd = 1.0 + u_prime * u_prime;

        // update Lorentz factor
        let tau = alpha * SPEED_OF_LIGHT * B;
        let u_star = u_prime * tau;
        let sigma = gamma_prime_sqd - tau * tau;

        let gamma = (
            0.5 * sigma +
            (0.25 * sigma.powi(2) + tau * tau + u_star.powi(2)).sqrt()
        ).sqrt();
        
        /*
        if gamma < 1.0 {
            println!("gamma = {}, {:?}", gamma, self);
        }
        assert!(gamma >= 1.0);
        */
        self.gamma_m1 = if gamma > 1.0 {gamma - 1.0} else {1.0};

        // and momentum
        let t = tau / (1.0 + self.gamma_m1);
        let s = 1.0 / (1.0 + t * t);

        self.u = s * (u_prime + (u_prime * t) * t + u_prime.cross(t));

        // then the position
        self.prev_x = self.x;
        let dxi = SPEED_OF_LIGHT * self.u.x * dt / (dx * (1.0 + self.gamma_m1));
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
}

impl Ion {
    pub fn with_charge_state(&mut self, charge_state: f64, atomic_number: f64, mass_number: f64) -> &mut Self {
        self.Z_star = charge_state;
        self.Z = atomic_number;
        self.A = mass_number;
        self
    }
}