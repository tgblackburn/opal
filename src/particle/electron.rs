use std::fmt;
use std::cmp::Ordering;

use mpi::traits::*;
use mpi::datatype::UserDatatype;
use memoffset::*;

use crate::constants::*;
use crate::particle::*;
use crate::particle::vec3::*;
use crate::qed::photon_emission;

#[derive(Copy,Clone)]
pub struct Electron {
    cell: isize,
    prev_x: f64,
    x: f64,
    y: f64,
    z: f64,
    weight: f64,
    gamma: f64,
    u: Vec3,
    chi: f64,
    tau: f64,
    work: f64,
    flag: bool,
}

impl fmt::Debug for Electron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[electron: x = {} + {}, gamma = {}, p/(mc) = {:?}]", self.cell, self.x, self.gamma, self.u)
    }
}

unsafe impl Equivalence for Electron {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        let blocklengths = [1; 12];
        let displacements = [
            offset_of!(Electron, cell) as mpi::Address,
            offset_of!(Electron, prev_x) as mpi::Address,
            offset_of!(Electron, x) as mpi::Address,
            offset_of!(Electron, y) as mpi::Address,
            offset_of!(Electron, z) as mpi::Address,
            offset_of!(Electron, weight) as mpi::Address,
            offset_of!(Electron, gamma) as mpi::Address,
            offset_of!(Electron, u) as mpi::Address,
            offset_of!(Electron, chi) as mpi::Address,
            offset_of!(Electron, tau) as mpi::Address,
            offset_of!(Electron, work) as mpi::Address,
            offset_of!(Electron, flag) as mpi::Address,
        ];
        let types: [&dyn Datatype; 12] = [
            &isize::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &Vec3::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &bool::equivalent_datatype(),
        ];
        UserDatatype::structured(12, &blocklengths, &displacements, &types)
    }
}

impl PartialOrd for Electron {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cell.cmp(&other.cell))
    }
}

impl PartialEq for Electron {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}

#[allow(non_snake_case)]
impl Particle for Electron {
    fn create(cell: isize, x: f64, u: &[f64; 3], weight: f64, dx: f64, dt: f64) -> Electron {
        let u = Vec3::new_from_slice(u);
        let gamma = (1.0 + u * u).sqrt();
        let prev_x = x - SPEED_OF_LIGHT * u.x * dt / (gamma * dx);
        Electron {
            cell: cell,
            prev_x: prev_x,
            x: x,
            y: 0.0,
            z: 0.0,
            weight: weight,
            gamma: gamma,
            u: u,
            chi: 0.0,
            tau: std::f64::INFINITY,
            work: 0.0,
            flag: false,
        }
    }

    fn location(&self) -> (isize, f64, f64) {
        (self.cell, self.x, self.prev_x)
    }

    fn transverse_displacement(&self) -> f64 {
        self.y.hypot(self.z)
    }

    #[inline]
    fn push(&mut self, E: &[f64; 3], B: &[f64; 3], dx: f64, dt: f64) {
        self.vay_push(E, B, dx, dt)
    }

    fn energy(&self) -> f64 {
        self.gamma * ELECTRON_MASS_MEV
    }

    fn total_kinetic_energy(&self) -> f64 {
        // gamma - 1 = (p/m)^2 / (gamma + 1)
        let to_joules = 1.0e6 * ELECTRON_MASS_MEV * ELEMENTARY_CHARGE;
        self.weight * self.u.norm_sqr() * to_joules / (self.gamma + 1.0)
    }

    fn work(&self) -> f64 {
        self.work // in joules
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn shift_cell(&mut self, delta: isize) {
        self.cell = self.cell + delta;
    }

    fn charge(&self) -> f64 {
        ELECTRON_CHARGE
    }

    fn mass(&self) -> f64 {
        ELECTRON_MASS
    }

    fn velocity(&self) -> [f64; 3] {
        let v = SPEED_OF_LIGHT * self.u / self.gamma;
        [v.x, v.y, v.z]
    }

    fn chi(&self) -> f64 {
        self.chi
    }

    fn momentum(&self) -> [f64; 3] {
        let p = self.u * ELECTRON_MASS_MEV;
        [p.x, p.y, p.z]
    }

    fn with_optical_depth(&self, tau: f64) -> Self {
        let mut pt = *self;
        pt.tau = tau;
        pt
    }

    fn flag(&mut self) {
        self.flag = true;
    }

    fn unflag(&mut self) {
        self.flag = false;
    }

    fn is_flagged(&self) -> bool {
        self.flag
    }

    fn normalized_four_momentum(&self) -> [f64; 4] {
        [self.gamma, self.u.x, self.u.y, self.u.z]
    }
}

#[allow(unused)]
impl Electron {
    /// If the electron optical depth has fallen below zero,
    /// emit a photon, recoil, and reset the optical depth.
    /// 
    /// Returns the photon and its estimated formation length
    /// (in metres), if emission has taken place.
    /// 
    /// The photon momentum is sampled from the relevant
    /// synchrotron spectrum using the supplied pseudorandom
    /// number generator.
    pub fn radiate<R: Rng>(&mut self, rng: &mut R, t: f64) -> Option<(Photon, f64)> {
        if self.tau < 0.0 {
            // reset optical depth against emission
            self.tau = rng.sample(Exp1);

            // determine photon energy and emission angle
            let (omega_mc2, theta, cphi) = if cfg!(feature = "no_radiation_reaction") {
                photon_emission::classical_sample(self.chi, self.gamma, rng.gen(), rng.gen(), rng.gen())
            } else {
                photon_emission::sample(self.chi, self.gamma, rng.gen(), rng.gen(), rng.gen())
            };

            // get unit vectors parallel and perp to electron momentum
            let parallel: Vec3 = self.u.normalize();
            let perp: Vec3 = parallel.orthogonal();
            let perp = perp.rotate_around(parallel, cphi);
            let u = if cfg!(feature = "no_beaming") {
                omega_mc2 * parallel
            } else {
                omega_mc2 * (theta.cos() * parallel + theta.sin() * perp)
            };

            // estimate photon formation length
            let formation_length = 2.0 * self.gamma.powi(2) * theta * SPEED_OF_LIGHT * COMPTON_TIME / self.chi;

            // electron recoils
            if cfg!(not(feature = "no_radiation_reaction")) {
                self.u = self.u - u;
                let new_gamma = (1.0 + self.u * self.u).sqrt();
                self.chi = self.chi * new_gamma / self.gamma;
                self.gamma = new_gamma;
            }
            
            // construct photon
            let u = [u.x, u.y, u.z];
            let photon = Photon::create(self.cell, self.x, &u, self.weight, 0.0, 0.0)
                .with_optical_depth(rng.sample(Exp1))
                .at_time(t);

            Some((photon, formation_length))
        } else {
            None
        }
    }

    /// Modifies the electron momentum by absorbing the specified
    /// photon.
    pub fn absorb(&mut self, photon: &Photon) {
        let k = photon.momentum(); // in units of MeV
        let scale = photon.weight() / self.weight;
        self.u.x += scale * k[0] / ELECTRON_MASS_MEV;
        self.u.y += scale * k[1] / ELECTRON_MASS_MEV;
        self.u.z += scale * k[2] / ELECTRON_MASS_MEV;
        self.gamma = (1.0 + self.u.norm_sqr()).sqrt();
    }

    /// Advances the particle momentum and position using
    /// the leapfrog pusher developed by Vay et al.,
    /// see https://doi.org/10.1063/1.2837054.
    #[allow(non_snake_case)]
    pub fn vay_push(&mut self, E: &[f64; 3], B: &[f64; 3], dx: f64, dt: f64) {
        let E = Vec3::new_from_slice(E);
        let B = Vec3::new_from_slice(B);
        
        // velocity in SI units
        let v = SPEED_OF_LIGHT * self.u / self.gamma;

        // u_i = u_{i-1/2} + (q dt/2 m c) (E + v_{i-1/2} x B)
        let alpha = ELECTRON_CHARGE * dt / (2.0 * ELECTRON_MASS * SPEED_OF_LIGHT);
        let u_half = self.u + alpha * (E + v.cross(B));
        self.work += ELECTRON_CHARGE * SPEED_OF_LIGHT * (u_half * E) * dt / (1.0 + u_half * u_half).sqrt();
        
        // quantum parameter
        let gamma_half = (1.0 + u_half * u_half).sqrt();
        self.chi = ((gamma_half * E + SPEED_OF_LIGHT * u_half.cross(B)).norm_sqr() - (E * u_half).powi(2)).sqrt() / CRITICAL_FIELD;

        if cfg!(feature = "no_radiation_reaction") {
            self.tau = self.tau - photon_emission::classical_rate(self.chi, gamma_half) * dt;
        } else {
            self.tau = self.tau - photon_emission::rate(self.chi, gamma_half) * dt;
        }
        
        // u' =  u_{i-1/2} + (q dt/2 m c) (2 E + v_{i-1/2} x B)
        let u_prime = u_half + alpha * E;
        let gamma_prime_sqd = 1.0 + u_prime * u_prime;

        // update Lorentz factor
        let tau = alpha * SPEED_OF_LIGHT * B;
        let u_star = u_prime * tau;
        let sigma = gamma_prime_sqd - tau * tau;

        self.gamma = (
            0.5 * sigma +
            (0.25 * sigma.powi(2) + tau * tau + u_star.powi(2)).sqrt()
        ).sqrt();

        // and momentum
        let t = tau / self.gamma;
        let s = 1.0 / (1.0 + t * t);

        self.u = s * (u_prime + (u_prime * t) * t + u_prime.cross(t));

        // then the position
        self.prev_x = self.x;
        let dxi = SPEED_OF_LIGHT * self.u.x * dt / (dx * self.gamma);
        assert!(dxi < 1.0);
        self.x = self.x + dxi;
        self.y = self.y + v.y * dt;
        self.z = self.z + v.z * dt;

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

    /// Advances the particle momentum and position using
    /// the Boris push.
    #[allow(non_snake_case)]
    pub fn boris_push(&mut self, E: &[f64; 3], B: &[f64; 3], dx: f64, dt: f64) {
        let E = Vec3::new_from_slice(E);
        let cB = SPEED_OF_LIGHT * Vec3::new_from_slice(B);

        let q = ELECTRON_CHARGE;
        let M = ELECTRON_MASS;
        let alpha = q * dt / (2.0 * M * SPEED_OF_LIGHT);

        // half the electric field acceleration:
        // u_ = u + alpha E
        let u_minus = self.u + alpha * E;

        // magnetic field rotation:
        // u' = u_ + t (u_ x cB)
        let gamma = 1.0 + u_minus.norm_sqr() / (1.0 + (1.0 + u_minus.norm_sqr()).sqrt());
        let t = alpha / gamma;
        let u_prime = u_minus + t * u_minus.cross(cB);

        // u+ = u_ + t' (u' x B)
        let t_prime = 2.0 * t / (1.0 + t.powi(2) * cB.norm_sqr());
        let u_plus = u_minus + t_prime * u_prime.cross(cB);

        // quantum parameter
        self.chi = ((gamma * E + u_plus.cross(cB)).norm_sqr() - (E * u_plus).powi(2)).sqrt() / CRITICAL_FIELD;

        if cfg!(feature = "no_radiation_reaction") {
            self.tau = self.tau - photon_emission::classical_rate(self.chi, gamma) * dt;
        } else {
            self.tau = self.tau - photon_emission::rate(self.chi, gamma) * dt;
        }

        // remaining electric field acceleration
        // u = u+ + alpha E
        self.u = u_plus + alpha * E;
        self.gamma = (1.0 + self.u.norm_sqr()).sqrt();

        // then the position
        self.prev_x = self.x;
        let v = SPEED_OF_LIGHT * self.u / self.gamma;
        let dxi = v.x * dt / dx;
        assert!(dxi < 1.0);
        self.x = self.x + dxi;
        self.y = self.y + v.y * dt;
        self.z = self.z + v.z * dt;

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

#[cfg(test)]
mod tests {
    use std::f64::consts;
    use super::*;

    #[test]
    fn static_magnetic_field() {
        let b0 = 1.0;
        let u0 = 1.0;
        let r_c = ELECTRON_MASS * SPEED_OF_LIGHT * u0 / (ELECTRON_CHARGE.abs() * b0);
        let omega_c = SPEED_OF_LIGHT * u0 / ((1.0 + u0 * u0).sqrt() * r_c);
        println!("r_c = {}, omega_c = {}", r_c, omega_c);
        let dt = 0.01 * 2.0 * consts::PI / omega_c;
        let mut e = Electron::create(0, 0.0, &[u0, 0.0, 0.0], 1.0, 1.0, dt);
        for _i in 0..100 {
            e.push(&[0.0, 0.0, 0.0], &[0.0, 0.0, b0], 1.0, dt);
        }
        let target = Electron::create(0, 0.0, &[1.0, 0.0, 0.0], 1.0, 1.0, dt);
        println!("Expected {:?}", target);
        println!("Got {:?}", e);
        println!("Work done / (mc^2) = {}", e.work() / (ELECTRON_MASS * SPEED_OF_LIGHT_SQD));
        assert!( (e.gamma - target.gamma) / target.gamma < 1.0e-6 )
        //assert!( e.x().abs() < 1.0e-2 * r_c )
    }

    /// Analytical solution to motion of charge, initially
    /// at rest in a static electric field Ex is
    ///     gamma(t) = [1 + (e Ex t / m c)^2]^(1/2)
    #[test]
    fn static_electric_field() {
        let e0 = 1.0e3;
        let dt = 1.0e-9;
        let mut e = Electron::create(0, 0.0, &[0.0, 0.0, 0.0], 1.0, 1.0, dt);
        for _i in 0..100 {
            e.push(&[e0, 0.0, 0.0], &[0.0, 0.0, 0.0], 1.0, dt);
        }
        let target = (1.0 + (ELECTRON_CHARGE * e0 * 1.0e-7 / (ELECTRON_MASS * SPEED_OF_LIGHT)).powi(2)).sqrt();
        println!("Expected gamma = {}, got {}", target, e.gamma);
        println!("{:?}", e);
        println!("Work done / (mc^2) = {}", e.work() / (ELECTRON_MASS * SPEED_OF_LIGHT_SQD));
        assert!( (e.gamma - target) / target < 1.0e-6 )
    }
}
