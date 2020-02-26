//! Custom implementations of special functions not
//! provided by the standard lib.

mod airy;

pub use airy::*;

const SERIES_MAX_LENGTH: usize = 20;

/// Represents the series expansion of a function
/// in powers of its dependent variable,
/// i.e. f(x) ≈ Σ_i a[i] x^(n[i]).
/// The powers n[i] can either be floating-point numbers
/// or integers.
struct Series<T> {
    a: [f64; SERIES_MAX_LENGTH],
    n: [T; SERIES_MAX_LENGTH],
}

impl Series<i32> {
    /// Returns the value of series expansion at `x`
    fn evaluate_at(&self, x: f64) -> f64 {
        self.a.iter()
            .zip(self.n.iter())
            .map(|(a, p)| a * x.powi(*p))
            .sum::<f64>()
    }
}

impl Series<f64> {
    /// Returns the value of series expansion at `x`
    fn evaluate_at(&self, x: f64) -> f64 {
        self.a.iter()
            .zip(self.n.iter())
            .map(|(a, p)| a * x.powf(*p))
            .sum::<f64>()
    }
}