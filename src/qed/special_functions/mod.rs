mod airy;

pub use airy::*;

const SERIES_MAX_LENGTH: usize = 20;

struct Series<T> {
    a: [f64; SERIES_MAX_LENGTH],
    n: [T; SERIES_MAX_LENGTH],
}

impl Series<i32> {
    fn evaluate_at(&self, x: f64) -> f64 {
        self.a.iter()
            .zip(self.n.iter())
            .map(|(a, p)| a * x.powi(*p))
            .sum::<f64>()
    }
}

impl Series<f64> {
    fn evaluate_at(&self, x: f64) -> f64 {
        self.a.iter()
            .zip(self.n.iter())
            .map(|(a, p)| a * x.powf(*p))
            .sum::<f64>()
    }
}