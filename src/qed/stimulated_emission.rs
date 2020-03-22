//! Stimulated photon emission: gamma + e -> e + gamma + gamma in a background field

use std::f64::consts;

use crate::constants::*;
use super::special_functions::*;

const CLASSICAL_ELECTRON_RADIUS: f64 = 2.81794e-15;

/// The scaled cross section for stimulated emission is given by
///   scaled_sigma = k.p sigma / (k0 p0)
/// where k, p are the four-momenta of the photon and electron
/// respectively, normalized to the electron mass.
/// 
/// The probability that a photon stimulates the emission of
/// another photon of the same momentum is given by
///   P = w (c dt / dx) scaled_sigma
/// where we assume that both particles occupy a volume of
/// A dx, and A = 1 in 1D. w is the weight of the macrophoton.
pub fn scaled_cross_section(k: [f64; 4], p: [f64; 4], chi_gamma: f64, chi_e: f64) -> Option<f64> {
    // Electron cannot emit a photon with more energy than itself!
    if chi_gamma >= chi_e || k[0] >= p[0] || chi_e <= 0.0 || chi_gamma <= 0.0 {
        return None;
    }

    let g = 0.5 + 0.25 * chi_gamma.powi(2) / (chi_e * (chi_e - chi_gamma));
    let z = (chi_gamma / (chi_e * (chi_e - chi_gamma))).powf(2.0/3.0);
    let k_p = k[0] * p[0] - k[1] * p[1] - k[2] * p[2] - k[3] * p[3];
    let zbar = 2.0 * z * chi_e * k_p / chi_gamma;

    let zbar_z = 2.0 * p[0] * k_p / k[0]; // guarantees positivity of cross section
    // let zbar_z = 2.0 * chi_e * k_p / chi_gamma;

    if let Some(ai) = airy_ai_for_positive(zbar) {
        let sigma = (2.0 * consts::PI * CLASSICAL_ELECTRON_RADIUS).powi(2) * chi_e * z * (4.0 * g * zbar_z - 1.0) * ai / (ALPHA_FINE * chi_gamma * k[0] * p[0]);
        Some(sigma)
    } else {
        None
    }
}