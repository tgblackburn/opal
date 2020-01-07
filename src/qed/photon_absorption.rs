use std::f64::consts;
use libc::{c_int, c_double, c_char};

use crate::constants::*;

const CLASSICAL_ELECTRON_RADIUS: f64 = 2.81794e-15;

#[allow(unused)]
#[repr(C)]
enum gsl_mode {
    PrecDouble,
    PrecSingle,
    PrecApprox,
}

#[repr(C)]
struct gsl_result {
    val: f64,
    err: f64,
}

type GslErrorHandler = extern fn(*const c_char, *const c_char, c_int, c_int);

#[link(name = "gsl")]
#[link(name = "gslcblas")]
extern {
    fn gsl_sf_airy_Ai_e(x: c_double, mode: gsl_mode, result: *mut gsl_result) -> c_int;
    fn gsl_set_error_handler_off() -> GslErrorHandler;
    //fn gsl_set_error_handler(handler: GslErrorHandler) -> GslErrorHandler;
}

fn airy_ai(z: f64) -> Option<f64> {
    let mode = gsl_mode::PrecDouble;
    let mut result = gsl_result {val: 0.0, err: 0.0};
    let status = unsafe {
        gsl_sf_airy_Ai_e(z, mode, &mut result)
    };
    if status == 0 {
        Some(result.val)
    } else {
        None
    }
}

pub fn disable_gsl_abort_on_error() {
    unsafe {
        gsl_set_error_handler_off();
    }
}

/// The scaled cross section for one-photon absorption is given by
///   scaled_sigma = k.p sigma / (k0 p0)
/// where k, p are the four-momenta of the photon and electron
/// respectively, normalized to the electron mass.
/// 
/// The probability for a photon to be absorbed by an electron is
///   P = w (c dt / dx) scaled_sigma
/// where we assume that both particles occupy a volume of
/// A dx, and A = 1 in 1D. w is the weight of the macrophoton.
pub fn scaled_cross_section(k: [f64; 4], p: [f64; 4], chi_gamma: f64, chi_e: f64) -> Option<f64> {
    if chi_e <= 0.0 || chi_gamma <= 0.0 {
        return None;
    }
    let g = 0.5 + 0.25 * chi_gamma.powi(2) / (chi_e * (chi_e + chi_gamma));
    let z = (chi_gamma / (chi_e * (chi_e + chi_gamma))).powf(2.0/3.0);
    let k_p = k[0] * p[0] - k[1] * p[1] - k[2] * p[2] - k[3] * p[3];
    let zbar = 2.0 * z * chi_e * k_p / chi_gamma;

    let zbar_z = 2.0 * p[0] * k_p / k[0]; // guarantees positivity of cross section
    // let zbar_z = 2.0 * chi_e * k_p / chi_gamma;

    if let Some(ai) = airy_ai(zbar) {
        let sigma = (2.0 * consts::PI * CLASSICAL_ELECTRON_RADIUS).powi(2) * chi_e * z * (4.0 * g * zbar_z - 1.0) * ai / (ALPHA_FINE * chi_gamma * k[0] * p[0]);
        Some(sigma)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn airy_0() {
        let val = airy_ai(0.0).unwrap();
        let target = 0.355028053888;
        println!("Ai(0) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < 1.0e-9 );
    }

    #[test]
    fn airy_2() {
        let val = airy_ai(2.0).unwrap();
        let target = 0.0349241304233;
        println!("Ai(2) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < 1.0e-9 );
    }

    #[test]
    fn airy_20() {
        let val = airy_ai(20.0).unwrap();
        let target = 1.69167286867e-27;
        println!("Ai(20) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < 1.0e-9 );
    }

    #[test]
    #[should_panic]
    fn airy_200() {
        let _val = airy_ai(200.0).unwrap();
    }
}