#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct u4 {
    bits: u8,
}

// impl u4 {
//     pub fn from(bits: u8) -> Self {
//         Self {
//             bits: bits & 0x0f_u8,
//         }
//     }
// }

impl From<u8> for u4 {
    fn from(bits: u8) -> Self {
        Self {
            bits: bits & 0x0f_u8,
        }
    }
}
use std::ops::Add;
use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::BitXor;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Not;
use std::ops::Shl;
use std::ops::Shr;
use std::ops::Sub;

impl Add for u4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            bits: (self.bits + other.bits) & 0x0f_u8,
        }
    }
}
impl Mul for u4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            bits: (self.bits * other.bits) & 0x0f_u8,
        }
    }
}
impl Div for u4 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            bits: (self.bits / other.bits) & 0x0f_u8,
        }
    }
}

impl Not for u4 {
    type Output = Self;

    fn not(self) -> Self {
        Self {
            bits: (!self.bits) & 0x0f_u8,
        }
    }
}

impl Sub for u4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let rhs = !other + Self { bits: 0x01 };
        self + rhs
    }
}

impl BitAnd for u4 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self {
            bits: (self.bits & other.bits) & 0x0f_u8,
        }
    }
}

impl BitOr for u4 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self {
            bits: (self.bits | other.bits) & 0x0f_u8,
        }
    }
}

impl BitXor for u4 {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        Self {
            bits: (self.bits ^ other.bits) & 0x0f_u8,
        }
    }
}

impl Shl for u4 {
    type Output = Self;

    fn shl(self, other: Self) -> Self {
        Self {
            bits: (self.bits << (other.bits & 0x03_u8)) & 0x0f_u8,
        }
    }
}

impl Shr for u4 {
    type Output = Self;

    fn shr(self, other: Self) -> Self {
        Self {
            bits: (self.bits >> (other.bits & 0x03_u8)) & 0x0f_u8,
        }
    }
}

use std::fmt;

impl fmt::Display for u4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.bits, f)
    }
}
impl fmt::Binary for u4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&self.bits, f)
    }
}
impl fmt::LowerHex for u4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.bits, f)
    }
}
impl fmt::UpperHex for u4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.bits, f)
    }
}
