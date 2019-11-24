use std::fmt;

use mpi::traits::*;
use mpi::datatype::UserDatatype;

#[derive(Copy,Clone)]
#[repr(C)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

unsafe impl Equivalence for Vec3 {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        let mpi_double = f64::equivalent_datatype();
        UserDatatype::contiguous(3, &mpi_double)
    }
}

// Operator overloading

// Add two vectors together
impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {x: self.x + other.x, y: self.y + other.y, z: self.z + other.z}
    }
}

// Multiply, i.e. dot, two vectors
impl std::ops::Mul for Vec3 {
    type Output = f64;
    fn mul(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

// Multiply a vector by a scalar
impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f64) -> Vec3 {
        Vec3{x: self.x * other, y: self.y * other, z: self.z * other}
    }
}

// Multiply a scalar by a vector
impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3{x: self * other.x, y: self * other.y, z: self * other.z}
    }
}

// Negate a vector
impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        -1.0 * self
    }
}

// Divide a vector by a scalar
impl std::ops::Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f64) -> Vec3 {
        Vec3{x: self.x / other, y: self.y / other, z: self.z / other}
    }
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3{x: x, y: y, z: z}
    }

    pub fn new_from_slice(a: &[f64; 3]) -> Vec3 {
        Vec3{x: a[0], y: a[1], z: a[2]}
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm_sqr(self) -> f64 {
        self * self
    }

    pub fn normalize(self) -> Self {
        let mag = self.norm_sqr().sqrt();
        assert!(mag > 0.0);
        self / mag
    }
}