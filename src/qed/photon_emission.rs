use std::f64::consts;
use crate::constants::*;
use super::pwmci;

// columns of log(chi), log(h(chi))
// range is 0.01 <= chi <= 100
const DELTA_LN_CHI: f64 = 0.230258509299; // log(10)/10
static LN_H_CHI_TABLE: [[f64; 2]; 41] = [
    [-4.60517019e+0, 1.64660829e+0],
    [-4.37491168e+0, 1.64437994e+0],
    [-4.14465317e+0, 1.64162210e+0],
    [-3.91439466e+0, 1.63822181e+0],
    [-3.68413615e+0, 1.63404803e+0],
    [-3.45387764e+0, 1.62895132e+0],
    [-3.22361913e+0, 1.62276454e+0],
    [-2.99336062e+0, 1.61530492e+0],
    [-2.76310211e+0, 1.60637755e+0],
    [-2.53284360e+0, 1.59578044e+0],
    [-2.30258509e+0, 1.58331053e+0],
    [-2.07232658e+0, 1.56877061e+0],
    [-1.84206807e+0, 1.55197597e+0],
    [-1.61180957e+0, 1.53276058e+0],
    [-1.38155106e+0, 1.51098175e+0],
    [-1.15129255e+0, 1.48652339e+0],
    [-9.21034037e-1, 1.45929745e+0],
    [-6.90775528e-1, 1.42924408e+0],
    [-4.60517019e-1, 1.39633080e+0],
    [-2.30258509e-1, 1.36055104e+0],
    [0.00000000e0, 1.32192247e+0],
    [2.30258509e-1, 1.28048533e+0],
    [4.60517019e-1, 1.23630061e+0],
    [6.90775528e-1, 1.18944834e+0],
    [9.21034037e-1, 1.14002554e+0],
    [1.15129255e+0, 1.08814410e+0],
    [1.38155106e+0, 1.03392829e+0],
    [1.61180957e+0, 9.77512163e-1],
    [1.84206807e+0, 9.19036694e-1],
    [2.07232658e+0, 8.58647059e-1],
    [2.30258509e+0, 7.96489918e-1],
    [2.53284360e+0, 7.32710951e-1],
    [2.76310211e+0, 6.67452686e-1],
    [2.99336062e+0, 6.00852676e-1],
    [3.22361913e+0, 5.33042051e-1],
    [3.45387764e+0, 4.64144437e-1],
    [3.68413615e+0, 3.94275229e-1],
    [3.91439466e+0, 3.23541171e-1],
    [4.14465317e+0, 2.52040210e-1],
    [4.37491168e+0, 1.79861566e-1],
    [4.60517019e+0, 1.07085976e-1],
];

pub fn rate(chi: f64, gamma: f64) -> f64 {
    let h = if chi < 0.01 {
        5.0 * consts::FRAC_PI_3 * (1.0 - 8.0 * chi / (5.0 * 3.0f64.sqrt()))
    } else if chi >= 100.0 {
        let cbrt_chi = chi.cbrt();
        let mut h = -1019.4661473121777 + 1786.716527650374 * cbrt_chi * cbrt_chi;
        h = 1750.6263395722715 + cbrt_chi * cbrt_chi * h;
        h = -2260.1819695887225 + cbrt_chi * h;
        h = 0.00296527643253334 * h / (chi * chi);
        h
    } else {
        let index = (chi.ln() - LN_H_CHI_TABLE[0][0]) / DELTA_LN_CHI;
        let weight = index.fract(); // of upper entry
        let index = index.floor() as usize;
        assert!(index < LN_H_CHI_TABLE.len() - 1);
        let ln_h = (1.0 - weight) * LN_H_CHI_TABLE[index][1] + weight * LN_H_CHI_TABLE[index+1][1];
        ln_h.exp()
    };

    3.0f64.sqrt() * ALPHA_FINE * chi * h / (2.0 * consts::PI * gamma * COMPTON_TIME)
}

fn max_z_at(gamma: f64) -> f64 {
    (2.0 * gamma.powi(2) * (1.0 + (1.0 - gamma.powi(-2)).sqrt())).powf(1.5)
}

pub fn classical_sample(chi: f64, gamma: f64, rand1: f64, rand2: f64, rand3: f64) -> (f64, f64, f64) {
    // First determine z:
    // z^(1/3) = (2 + 4 cos(delta/3)) / (5 (1-r)) where 0 <= r < 1
    // and cos(delta) = (-9 + 50r - 25r^2) / 16
    let delta = ((-9.0 + 50.0 * rand1 - 25.0 * rand1.powi(2)) / 16.0).acos();
    let z = ((2.0 + 4.0 * (delta/3.0).cos()) / (5.0 * (1.0 - rand1))).powi(3);

    // verify that z <= z_max(gamma)
    let z = z.min(max_z_at(gamma));

    // now invert cdf(u|z) = (3/pi) \int_0^x t K_{1/3}(t) dt,
    // which is tabulated, to obtain x = 2 u z / (3 chi)
    // for x < 0.01, cdf(u|z) =

    let x = 1.020377255 * rand2.powf(0.6);
    /*
    let x = if rand2 < CLASSICAL_SPECTRUM_TABLE[0][1] {
        1.020377255 * rand2.powf(0.6)
    } else {
        pwmci::invert(rand2, &CLASSICAL_SPECTRUM_TABLE)
    };
    */
    let u = 3.0 * chi * x / (2.0 * z);
    let omega_mc2 = u * gamma;

    (omega_mc2, 0.0, 2.0 * consts::PI * rand3)
}

pub fn classical_rate(chi: f64, gamma: f64) -> f64 {
    let h = 5.0 * consts::FRAC_PI_3;
    3.0f64.sqrt() * ALPHA_FINE * chi * h / (2.0 * consts::PI * gamma * COMPTON_TIME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_0_026() {
        let value = rate(0.026, 1000.0);
        let target = 2.07935e14;
        println!("rate(chi = 0.026, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_3_5() {
        let value = rate(3.5, 1000.0);
        let target = 1.58485e16;
        println!("rate(chi = 3.5, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_9_98() {
        let value = rate(9.98, 1000.0);
        let target = 3.45844e16;
        println!("rate(chi = 9.98, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_12_4() {
        let value = rate(12.4, 1000.0);
        let target = 4.04647e16;
        println!("rate(chi = 12.4, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_403() {
        let value = rate(403.0, 1000.0);
        let target = 4.46834e17;
        println!("rate(chi = 403, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }
}