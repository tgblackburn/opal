//! Airy functions
//! 
//! Currently implemented:
//! 
//! * Airy function of the first kind, Ai(z), for real,
//!   positive argument.
//! 
//! Algorithms adapted from:
//! 
//! * A. Gil, J. Segura and N. M. Tenne,
//!   "Computing Complex Airy Functions by Numerical Quadrature",
//!   Numerical Algorithms 30, 11--23 (2002)

use super::Series;

/// Returns the value of the Airy function for real, positive `x`.
/// If `x` is negative, or the result would underflow, the
/// return value is None.
pub fn airy_ai_for_positive(x: f64) -> Option<f64> {
    if x < 0.0 {
        None
    } else if x < 1.0 {
        // Use Taylor series expansion
        Some(SMALL_X_EXPANSION.evaluate_at(x))
    } else if x < 10.0 {
        // Numerically integrate the integal representation
        // the Airy function using 40-point Gauss-Laguerre
        // quadrature.
        //
        // That representation is
        //   Ai(x) = a(x) \int_0^\infty f(t) w(t) dt
        // where the integrand
        //   f(t) = (2 + t/s)^(-1/6),
        // the weight function
        //   w(t) = t^(-1/6) exp(-t),
        // the scale factor
        //   a(x) = s^(-1/6) exp(-s) / (sqrt(pi) (48)^(1/6) Gamma(5/6))
        // and
        //   s = 2 x^(3/2) / 3.
        let s = 2.0 * x.powf(1.5) / 3.0;
        let a = 0.262183997088323 * s.powf(-1.0/6.0) * (-s).exp();
        let integral: f64 = GAUSS_LAGUERRE_NODES.iter()
            .zip(GAUSS_LAGUERRE_WEIGHTS.iter())
            .map(|(x, w)| w * (2.0 + x / s).powf(-1.0/6.0))
            .sum();
        Some(a * integral)
    } else {
        // if x > 10, calculate expansion of exp(2z^(3/2)/3) Ai(x)
        // around x = infinity, for scaled argument y = x / 10.0.
        // This sum is well-behaved even if y is very large.
        let y = x / 10.0;
        let total = SCALED_LARGE_Y_EXPANSION.evaluate_at(y);
        // The scaling factor, on the other hand, can be
        // too small to represent as an f64.
        let scale = (-2.0 * x.powf(1.5) / 3.0).exp();
        // If it does underflow, return None instead.
        let value = scale * total;
        if value.is_normal() {
            Some(value)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn airy_0() {
        let val = airy_ai_for_positive(0.0).unwrap();
        let target = 0.355028053888;
        println!("Ai(0) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < 1.0e-12 );
    }

    #[test]
    fn airy_2() {
        let val = airy_ai_for_positive(2.0).unwrap();
        let target = 0.0349241304233;
        println!("Ai(2) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < 1.0e-12 );
    }

    #[test]
    fn airy_17() {
        let val = airy_ai_for_positive(17.0).unwrap();
        let target = 7.05019729838861e-22;
        println!("Ai(17) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < 1.0e-12 );
    }

    #[test]
    fn airy_20() {
        let val = airy_ai_for_positive(20.0).unwrap();
        let target = 1.69167286867e-27;
        println!("Ai(20) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < 1.0e-12 );
    }

    #[test]
    #[should_panic]
    fn airy_200() {
        let _val = airy_ai_for_positive(200.0).unwrap();
    }
}

static SMALL_X_EXPANSION: Series<i32> = Series {
    a: [
		3.550280538878172e-1,
		-2.588194037928068e-1,
		5.917134231463621e-2,
		-2.156828364940057e-2,
		1.972378077154540e-3,
		-5.135305630809659e-4,
		2.739413996047973e-5,
		-5.705895145344065e-6,
		2.075313633369676e-7,
		-3.657625093169273e-8,
		9.882445873188934e-10,
		-1.524010455487197e-10,
		3.229557474898344e-12,
		-4.456170922477184e-13,
		7.689422559281773e-15,
		-9.645391607093472e-16,
		1.393011333203220e-17,
		-1.607565267848912e-18,
		1.984346628494615e-20,
		-2.126409084456233e-21,
    ],
    n: [
        0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28
    ],
};

static SCALED_LARGE_Y_EXPANSION: Series<f64> = Series {
    a: [
		1.586335590354182e-1,
		-5.225451665528175e-4,
		1.325388893850608e-5,
		-6.432400968771192e-7,
		4.629704976902856e-8,
		-4.422624646117476e-9,
		5.268868167616043e-10,
		-7.522524928854403e-11,
		1.251983844382659e-11,
		-2.380054640631914e-12,
		5.088155686888228e-13,
		-1.208285769002228e-13,
		3.155588236828597e-14,
		-8.988957437753233e-15,
		2.773608979586462e-15,
		-9.215558717755193e-16,
		3.280389764897847e-16,
		-1.245456023749605e-16,
		5.023838356464561e-17,
		-2.145585178693777e-17,
    ],
    n: [
        -0.25,
        -1.75,
        -3.25,
        -4.75,
        -6.25,
        -7.75,
        -9.25,
        -10.75,
        -12.25,
        -13.75,
        -15.25,
        -16.75,
        -18.25,
        -19.75,
        -21.25,
        -22.75,
        -24.25,
        -25.75,
        -27.25,
        -28.75,
    ],
};
 
static GAUSS_LAGUERRE_NODES: [f64; 40] = [
    2.838914179945677e-2,
    1.709853788600349e-1,
    4.358716783417705e-1,
    8.235182579130309e-1,
    1.334525432542274e+0,
    1.969682932064351e+0,
    2.729981340028599e+0,
    3.616621619161009e+0,
    4.631026110526541e+0,
    5.774851718305477e+0,
    7.050005686302187e+0,
    8.458664375132378e+0,
    1.000329552427494e+1,
    1.168668459477224e+1,
    1.351196593446936e+1,
    1.548265969593771e+1,
    1.760271568080691e+1,
    1.987656560227855e+1,
    2.230918567739628e+1,
    2.490617202129742e+1,
    2.767383207394972e+1,
    3.061929632950841e+1,
    3.375065608502399e+1,
    3.707713497083912e+1,
    4.060930496943413e+1,
    4.435936195160668e+1,
    4.834148224345283e+1,
    5.257229170785049e+1,
    5.707149458398093e+1,
    6.186273503855476e+1,
    6.697480787736505e+1,
    7.244341162998353e+1,
    7.831377964843565e+1,
    8.464480548222756e+1,
    9.151587398018528e+1,
    9.903899485517280e+1,
    1.073824762956655e+2,
    1.168236917656583e+2,
    1.278937448431646e+2,
    1.419607885990635e+2,
];

static GAUSS_LAGUERRE_WEIGHTS: [f64; 40] = [
    1.437204088033139e-1,
    2.304075592418809e-1,
    2.422530455213276e-1,
    2.036366391034408e-1,
    1.437606306229214e-1,
    8.691288347060781e-2,
    4.541750018329159e-2,
    2.061180312060695e-2,
    8.142788212686070e-3,
    2.802660756633776e-3,
    8.403374416217193e-4,
    2.193037329077657e-4,
    4.974016590092524e-5,
    9.785080959209777e-6,
    1.665428246036952e-6,
    2.445027367996577e-7,
    3.085370342362143e-8,
    3.332960729372821e-9,
    3.067818923653773e-10,
    2.393313099090116e-11,
    1.572947076762871e-12,
    8.649360130178674e-14,
    3.948198167006651e-15,
    1.482711730481083e-16,
    4.533903748150563e-18,
    1.115479804520358e-19,
    2.177666605892262e-21,
    3.318788910579756e-23,
    3.872847904397466e-25,
    3.381185924262449e-27,
    2.146990618932626e-29,
    9.574538399305471e-32,
    2.868778345026473e-34,
    5.452034672917572e-37,
    6.082128006541067e-40,
    3.571351222207245e-43,
    9.375169717620775e-47,
    8.418177761921027e-51,
    1.554777624272071e-55,
    1.625726581852354e-61,
];