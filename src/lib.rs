use std::fmt::Write;

pub mod parser;

#[derive(Debug, Clone)]
pub struct BigInt {
    words: Vec<u8>,
}

pub use crate::parser::*;
pub use ParseBigIntError::*;

impl BigInt {
    pub fn from_rtol(rtol_words: &[u8]) -> Self {
        Self {
            words: rtol_words.iter().cloned().rev().collect(),
        }
    }

    pub fn try_from_hex(s: &str) -> Result<Self, ParseBigIntError> {
        if s.len() == 0 {
            Err(EmptyString)
        } else if !s.starts_with("0x") {
            Err(MissingPrefix)
        } else {
            let s = &s[2..];
            let n = s.len();
            if n == 0 {
                Err(ZeroDigits)
            } else {
                // `m` is perhaps an overestimate, given underscores are permitted,
                // but we need something to amortize the allocation.
                let m = n / 2 + n & 1;
                let mut words: Vec<u8> = Vec::with_capacity(m);
                let mut iter = s.chars().rev().filter(|c| *c != '_');
                let mut w: u8 = 0x00;
                let mut i: usize = 0;
                while let Some(c) = iter.next() {
                    if !is_hex_digit(c) {
                        return Err(InvalidDigit);
                    } else {
                        let u = char_to_hex(c);
                        if i & 1 == 1 {
                            words.push((u << 4) | w);
                        } else {
                            w = u;
                        }
                    }
                    i += 1;
                }
                if i == 0 {
                    return Err(NoValidDigits);
                }
                if i & 1 == 1 {
                    words.push(w);
                }
                words.shrink_to_fit();
                Ok(BigInt { words })
            }
        }
    }

    pub fn is_even(&self) -> bool {
        self.words[0] & 1 == 0
    }
    pub fn is_odd(&self) -> bool {
        self.words[0] & 1 == 1
    }
    pub fn is_zero(&self) -> bool {
        // This is the literal implementation
        // self.words.iter().all(|x| *x == 0)
        // This might be faster, depending on the internals of `all`
        self.words.iter().fold(0x00, |acc, x| acc | *x) == 0x00
    }

    pub fn halve(&mut self) {
        let mut msb: u8 = 0x00;
        for w in self.words.iter_mut().rev() {
            let t = *w;
            let w_prime = msb | (t >> 1);
            msb = t << 7;
            *w = w_prime;
        }
    }

    pub fn double(&mut self) {
        let mut lsb: u8 = 0x00;
        for w in self.words.iter_mut() {
            let t = *w;
            let w_prime = (t << 1) | lsb;
            lsb = t >> 7;
            *w = w_prime;
        }
        if lsb == 0x01 {
            self.words.push(0x01);
        }
    }

    pub fn decrement(&mut self) {
        // This assumes that the bigint is >= 1
        assert!(!self.is_zero(), "attempt to subtract from 0");
        let n = self.words.len();
        // Initial iteration; k is 1 initially, but write as const.
        let u_0 = self.words[0];
        let w_0 = u_0.wrapping_sub(1).wrapping_add(1).wrapping_add(u8::MAX);
        let mut k = (!(w_0 > u_0)) as u8;
        self.words[0] = w_0;
        // Then, loop as usual.
        let mut j: usize = 1;
        while j < n {
            let u_j = self.words[j];
            let w_j = u_j.wrapping_add(k).wrapping_add(u8::MAX);
            k = (!(w_j > u_j)) as u8;
            self.words[j] = w_j;
            j += 1;
        }
    }

    pub fn increment(&mut self) {
        let n = self.words.len();
        // Initial iteration; k is 0 initially.
        let u_j = self.words[0];
        let w_j = u_j.wrapping_add(1);
        let mut k = (w_j < u_j.max(1)) as u8;
        self.words[0] = w_j;
        // Then, loop as usual.
        let mut j: usize = 1;
        while j < n {
            let w_j = self.words[j];
            let w_j_prime = w_j.wrapping_add(k);
            k = (w_j_prime < w_j) as u8;
            self.words[j] = w_j_prime;
            j += 1;
        }
        if k != 0 {
            self.words.push(k);
        }
    }

    pub fn print_binary(&self) {
        print!("0b");
        self.words.iter().rev().for_each(|w| {
            print!("{:08b}", w);
        });
        print!("\n");
    }

    pub fn print_lower_hex(&self) {
        print!("0x");
        self.words.iter().rev().for_each(|w| {
            print!("{:02x}", w);
        });
        print!("\n");
    }

    pub fn to_lower_hex(&self) -> String {
        let mut s = String::from("0x");
        self.words.iter().rev().for_each(|w| {
            write!(&mut s, "{:02x}", w).unwrap();
        });
        s
    }
    pub fn to_upper_hex(&self) -> String {
        let mut s = String::from("0x");
        self.words.iter().rev().for_each(|w| {
            write!(&mut s, "{:02X}", w).unwrap();
        });
        s
    }
}

pub fn algorithm_a(u: &BigInt, v: &BigInt) -> BigInt {
    let n = u.words.len();
    assert_eq!(n, v.words.len());

    let mut w: Vec<u8> = Vec::with_capacity(n + 1);
    let mut j: usize = 0;
    let mut k: u8 = 0;
    while j < n {
        let u_j = u.words[j];
        let v_j = v.words[j];
        let w_j = u_j.wrapping_add(v_j);
        let k_prime = w_j < u_j.max(v_j);
        let w_j_prime = w_j.wrapping_add(k);
        k = (k_prime | (w_j_prime < w_j)) as u8;
        w.push(w_j_prime);
        j += 1;
    }
    if k != 0 {
        w.push(k);
    }
    BigInt { words: w }
}

pub fn algorithm_a_sz(u: &BigInt, v: &BigInt) -> BigInt {
    let m = u.words.len();
    let n = v.words.len();
    if m < n {
        algorithm_a_sz(v, u)
    } else {
        let mut w: Vec<u8> = Vec::with_capacity(m + 1);
        let mut j: usize = 0;
        let mut k: u8 = 0;
        while j < n {
            let u_j = u.words[j];
            let v_j = v.words[j];
            let w_j = u_j.wrapping_add(v_j);
            let k_prime = w_j < u_j.max(v_j);
            let w_j_prime = w_j.wrapping_add(k);
            k = (k_prime | (w_j_prime < w_j)) as u8;
            w.push(w_j_prime);
            j += 1;
        }
        while j < m {
            let w_j = u.words[j];
            let w_j_prime = w_j.wrapping_add(k);
            k = (w_j_prime < w_j) as u8;
            w.push(w_j_prime);
            j += 1;
        }
        if k != 0 {
            w.push(k);
        }
        BigInt { words: w }
    }
}

pub fn algorithm_s(u: &BigInt, v: &BigInt) -> BigInt {
    let n = u.words.len();
    assert_eq!(n, v.words.len());
    assert!(algorithm_ge(u, v));

    let mut w: Vec<u8> = Vec::with_capacity(n);
    let mut j: usize = 0;
    let mut k: u8 = 1;
    while j < n {
        let u_j = u.words[j];
        let w_j = u_j
            .wrapping_sub(v.words[j])
            .wrapping_add(k)
            .wrapping_add(u8::MAX);
        k = (!(w_j > u_j)) as u8;
        w.push(w_j);
        j += 1;
    }
    BigInt { words: w }
}

pub fn algorithm_m(u: &BigInt, v: &BigInt) -> BigInt {
    let m = u.words.len();
    let n = v.words.len();
    let mut w: Vec<u8> = Vec::with_capacity(m + n - 1);
    w.resize(m + n, 0);
    let mut j: usize = 0;
    while j < n {
        let mut i: usize = 0;
        let mut k: u16 = 0;
        let v_j = v.words[j] as u16;
        while i < m {
            let t = (u.words[i] as u16) * v_j + (w[i + j] as u16) + k;
            let w_ij = t & 0x00ff; // t & (( 1u16 << 8) - 1);
            w[i + j] = w_ij as u8;
            k = t >> 8;
            i = i + 1;
        }
        w[j + m] = k as u8;
        j = j + 1;
    }
    BigInt { words: w }
}

use std::cmp::Ordering;
fn cmp_same_size(u: &[u8], v: &[u8]) -> Ordering {
    let m = u.len();
    assert_eq!(m, v.len());
    let mut i: usize = m;
    let mut state = Ordering::Equal;
    while i > 0 {
        i -= 1;
        if u[i] == v[i] {
            continue;
        } else if u[i] > v[i] {
            state = Ordering::Greater;
            break;
        } else {
            state = Ordering::Less;
            break;
        }
    }
    state
}

pub fn algorithm_cmp(u: &BigInt, v: &BigInt) -> Ordering {
    let m = u.words.len();
    let n = v.words.len();
    if m < n {
        let mut i: usize = n;
        while i > m {
            i -= 1;
            if v.words[i] != 0 {
                return Ordering::Less;
            }
        }
        cmp_same_size(&u.words[0..i], &v.words[0..i])
    } else if m == n {
        cmp_same_size(&u.words[..], &v.words[..])
    } else {
        // m > n
        let mut i: usize = m;
        while i > n {
            i -= 1;
            if u.words[i] != 0 {
                return Ordering::Greater;
            }
        }
        cmp_same_size(&u.words[0..i], &v.words[0..i])
    }
}

pub fn algorithm_lt(u: &BigInt, v: &BigInt) -> bool {
    match algorithm_cmp(u, v) {
        Ordering::Less => true,
        _ => false,
    }
}
pub fn algorithm_gt(u: &BigInt, v: &BigInt) -> bool {
    match algorithm_cmp(u, v) {
        Ordering::Greater => true,
        _ => false,
    }
}
pub fn algorithm_eq(u: &BigInt, v: &BigInt) -> bool {
    match algorithm_cmp(u, v) {
        Ordering::Equal => true,
        _ => false,
    }
}
pub fn algorithm_le(u: &BigInt, v: &BigInt) -> bool {
    !algorithm_gt(u, v)
}
pub fn algorithm_ge(u: &BigInt, v: &BigInt) -> bool {
    !algorithm_lt(u, v)
}
pub fn algorithm_ne(u: &BigInt, v: &BigInt) -> bool {
    !algorithm_eq(u, v)
}

pub fn half(u: &BigInt) -> BigInt {
    let mut u = u.clone();
    u.halve();
    u
}

pub fn double(u: &BigInt) -> BigInt {
    let mut u = u.clone();
    u.double();
    u
}

// This, however, is very slow due to the fact that the running
// time is now lg(min(u, v)), rather than a function of the
// number of words.
/*
A more rigorous analysis, based on u: m-place word, v: n-place word
where m >= n, in which a word consists of 2^p binary digits.
The Russian peasant algorithm requires lg((2^p)^n) = lg(2^(pn)) = pn steps.

The internal operations have time complexity:
Ω(m) : doubling, addition
Θ(n) : halving, decrement
Θ(n) : zero check
Θ(1) : even/odd check

Thus, we have the
∑ᵢ₌₀ᵖⁿ m + i + 2n + 1 = mpn + 2p * n^2 + ((p^2) * (n^2) + 3pn)/2

In the worst case, n=m, thus, 3p * n^2 + ((p^2) * (n^2) + 3pn)/2
which is O((p^2) * (n^2)), thus, we can determine an asymptotically tight
upper bound.
When m >= (n * (p + 4) + 3)/2, then the m-place word dominates the running time
of the algorithm. However, when this is not satisfied, the m-exclusive terms
will dominate, in which case the complexity equivalent to the worst case.
This enables us to determine an asymptotically tight lower bound, i.e. Ω(mpn),
which applies to all cases in which the m-derived terms dominate.

From the analysis, it is now clear that this algorithm is inferior to
the simple multiplication algorithm which has time complexity Θ(mn).
*/
pub fn algorithm_m_rp(u: &BigInt, v: &BigInt) -> BigInt {
    if algorithm_ge(u, v) {
        let mut a = BigInt { words: vec![0x00] };
        let mut b = u.clone();
        let mut n = v.clone();
        while !n.is_zero() {
            if n.is_even() {
                b.double();
                n.halve();
            } else {
                let t = algorithm_a_sz(&a, &b);
                a = t;
                n.decrement();
            }
        }
        a
    } else {
        algorithm_m_rp(v, u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition_works() {
        let u = BigInt { words: vec![0xff] };
        let v = BigInt { words: vec![0xff] };
        let w = algorithm_a(&u, &v);
        assert_eq!(w.words[0], 0xfe);
        assert_eq!(w.words[1], 0x01);
    }

    #[test]
    fn addition_varsize_works() {
        let u = BigInt::from_rtol(&[0x01]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0x00, 0x00, 0x01]);
        let w = algorithm_a_sz(&u, &v);
        assert_eq!(w.words.len(), 5);
        let rhs = BigInt::from_rtol(&[0x00, 0x00, 0x00, 0x00, 0x02]);
        assert!(algorithm_eq(&w, &rhs));

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x01]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0x00, 0x01]);
        let w = algorithm_a_sz(&u, &v);

        let lhs = BigInt::from_rtol(&[0x00, 0x00, 0x01, 0x01, 0x01]);
        let rhs = algorithm_a(&lhs, &v);
        assert!(algorithm_eq(&w, &rhs));

        let u = BigInt::from_rtol(&[0x0f, 0xf0, 0x01, 0x01, 0x01]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0x00, 0x01]);
        let w = algorithm_a_sz(&u, &v);

        let lhs = BigInt::from_rtol(&[0x0f, 0xf0, 0x01, 0x01, 0x01]);
        let rhs = algorithm_a(&lhs, &v);
        assert!(algorithm_eq(&w, &rhs));
    }

    #[test]
    fn subtraction_works() {
        // N.B. The right-to-left order of the positional number system
        // is reversed here, such that one writes the digits from left-to-right.
        // e.g. for a_2, a_1, a_0, one would write vec![a_0, a_1, a_2]
        let u = BigInt {
            words: vec![0x00, 0x01],
        };
        let v = BigInt {
            words: vec![0x01, 0x00],
        };

        let w = algorithm_s(&u, &v);
        // let rhs = BigInt {words: vec![0x00, 0xff]};
        assert_eq!(w.words[0], 0xff);
        assert_eq!(w.words[1], 0x00);
    }

    #[test]
    fn multiplication_works() {
        let u = BigInt { words: vec![0xff] };
        let v = BigInt { words: vec![0xff] };
        let w = algorithm_m(&u, &v);
        assert_eq!(w.words[0], 0x01);
        assert_eq!(w.words[1], 0xfe);

        let u = BigInt { words: vec![0x0f] };
        let v = BigInt { words: vec![0x0f] };
        let w = algorithm_m(&u, &v);
        assert_eq!(w.words[0], 0xe1);
        assert_eq!(w.words[1], 0x00);
    }

    #[test]
    fn multiplication_rp_works() {
        let u = BigInt::from_rtol(&[0xba, 0xdc, 0x0f, 0xfe, 0xe0]);
        let v = BigInt::from_rtol(&[0xba, 0xdf, 0x00, 0xd5, 0xee]);

        let lhs = algorithm_m(&u, &v);
        let rhs = algorithm_m_rp(&u, &v);

        assert!(algorithm_eq(&lhs, &rhs));
    }

    #[test]
    fn lt_works() {
        let u = BigInt::from_rtol(&[0x01, 0x00]);
        let v = BigInt::from_rtol(&[0x02, 0x01]);
        assert!(algorithm_lt(&u, &v));

        let w = BigInt::from_rtol(&[0x00, 0x00, 0x02, 0x01]);
        assert!(algorithm_lt(&u, &w));

        let x = BigInt::from_rtol(&[0x00, 0x01, 0x02, 0x01]);
        assert!(!algorithm_lt(&x, &v));

        let u = BigInt::from_rtol(&[0x01, 0xff]);
        let v = BigInt::from_rtol(&[0x02, 0x01]);
        assert!(algorithm_lt(&u, &v));

        let u = BigInt::from_rtol(&[0xff, 0x01]);
        let v = BigInt::from_rtol(&[0x02, 0xff]);
        assert!(!algorithm_lt(&u, &v));
    }

    #[test]
    fn gt_works() {
        let u = BigInt::from_rtol(&[0x01, 0x00]);
        let v = BigInt::from_rtol(&[0x02, 0x01]);
        assert!(algorithm_gt(&v, &u));

        let w = BigInt::from_rtol(&[0x00, 0x00, 0x02, 0x01]);
        assert!(algorithm_gt(&w, &u));

        let x = BigInt::from_rtol(&[0x00, 0x01, 0x02, 0x01]);
        assert!(!algorithm_gt(&v, &x));

        let u = BigInt::from_rtol(&[0x01, 0xff]);
        let v = BigInt::from_rtol(&[0x02, 0x01]);
        assert!(algorithm_gt(&v, &u));

        let u = BigInt::from_rtol(&[0xff, 0x01]);
        let v = BigInt::from_rtol(&[0x02, 0xff]);
        assert!(!algorithm_gt(&v, &u));
    }

    #[test]
    fn eq_works() {
        let u = BigInt::from_rtol(&[0x00, 0x00]);
        let v = BigInt::from_rtol(&[0x00]);
        assert!(algorithm_eq(&u, &v));

        let u = BigInt::from_rtol(&[0x00, 0x00, 0x00, 0x1f]);
        let v = BigInt::from_rtol(&[0x00, 0x1f]);
        assert!(algorithm_eq(&u, &v));

        let u = BigInt::from_rtol(&[0x00, 0x00, 0x00, 0x1f]);
        let v = BigInt::from_rtol(&[0x1f, 0x00]);
        assert!(!algorithm_eq(&u, &v));
    }

    #[test]
    fn is_even_works() {
        let u = BigInt::from_rtol(&[0x00]);
        assert!(u.is_even());

        let u = BigInt::from_rtol(&[0x01]);
        assert!(!u.is_even());

        let u = BigInt::from_rtol(&[0xfe]);
        assert!(u.is_even());
    }

    #[test]
    fn is_odd_works() {
        let u = BigInt::from_rtol(&[0x00]);
        assert!(!u.is_odd());

        let u = BigInt::from_rtol(&[0x03]);
        assert!(u.is_odd());

        let u = BigInt::from_rtol(&[0x04]);
        assert!(!u.is_odd());

        let u = BigInt::from_rtol(&[0xff]);
        assert!(u.is_odd());
    }

    #[test]
    fn is_zero_works() {
        let u = BigInt::from_rtol(&[0x00]);
        assert!(u.is_zero());

        let u = BigInt::from_rtol(&[0x01, 0x02, 0x03, 0x04, 0x05, 0xff]);
        assert!(!u.is_zero());

        let u = BigInt::from_rtol(&[0x00, 0x00, 0x00, 0x00, 0x00]);
        assert!(u.is_zero());
    }

    #[test]
    fn decrement_works() {
        let mut u = BigInt::from_rtol(&[0x01]);
        u.decrement();
        assert!(u.is_zero());

        let mut u = BigInt::from_rtol(&[0x01, 0x00]);
        u.decrement();
        let v = BigInt::from_rtol(&[0x00, 0xff]);
        assert!(algorithm_eq(&u, &v));
    }

    #[test]
    fn increment_works() {
        let mut u = BigInt::from_rtol(&[0x00]);
        let v = BigInt::from_rtol(&[0x00, 0x01]);
        u.increment();
        assert!(algorithm_eq(&u, &v));

        let mut u = BigInt::from_rtol(&[0x01, 0xff]);
        u.increment();
        let v = BigInt::from_rtol(&[0x02, 0x00]);
        assert!(algorithm_eq(&u, &v));
    }

    #[test]
    fn half_works() {
        let u = BigInt::from_rtol(&[0xff]);
        let v = algorithm_a(&u, &u);
        let w = half(&v);
        assert!(algorithm_eq(&u, &w));

        let u = BigInt::from_rtol(&[0xff]);
        let u2 = BigInt::from_rtol(&[0x02]);
        let v = algorithm_m(&u, &u2);
        let w = half(&v);
        assert!(algorithm_eq(&u, &w));
    }

    #[test]
    fn double_works() {
        let u = BigInt::from_rtol(&[0x00]);
        let d = double(&u);
        let rhs = BigInt::from_rtol(&[0x00]);
        assert!(algorithm_eq(&d, &rhs));

        let u = BigInt::from_rtol(&[0x01]);
        let d = double(&u);
        let rhs = BigInt::from_rtol(&[0x02]);
        assert!(algorithm_eq(&d, &rhs));

        let u = BigInt::from_rtol(&[0x0f]);
        let d = double(&u);
        let rhs = BigInt::from_rtol(&[0x1e]);
        assert!(algorithm_eq(&d, &rhs));

        let u = BigInt::from_rtol(&[0xff]);
        let d = double(&u);
        let rhs = BigInt::from_rtol(&[0x01, 0xfe]);
        assert!(algorithm_eq(&d, &rhs));

        let mut d = d;
        d.double();
        let rhs = BigInt::from_rtol(&[0x03, 0xfc]);
        assert!(algorithm_eq(&d, &rhs));

        let mut u = BigInt::from_rtol(&[0x01]);
        for p in 1..7 {
            u.double();
            let rhs = BigInt::from_rtol(&[1 << p]);
            assert!(algorithm_eq(&u, &rhs));
        }
    }

    #[test]
    fn to_lower_hex_works() {
        let u = BigInt::from_rtol(&[0x00]);
        assert_eq!(u.to_lower_hex(), String::from("0x00"));

        let u = BigInt::from_rtol(&[0x00, 0x01, 0xff]);
        assert_eq!(u.to_lower_hex(), String::from("0x0001ff"));
    }

    #[test]
    fn try_from_hex_works() {
        let s = "0xfff";

        let u = BigInt::try_from_hex(s).unwrap();
        let v = BigInt::from_rtol(&[0x0f, 0xff]);
        assert!(algorithm_eq(&u, &v));

        let s = "0xbadc0ffeebadcafe";
        let u = BigInt::try_from_hex(s).unwrap();
        let v = BigInt::from_rtol(&[0xba, 0xdc, 0x0f, 0xfe, 0xeb, 0xad, 0xca, 0xfe]);
        assert!(algorithm_eq(&u, &v));

        let s = "0xbad_c0ffee_bad_cafe";
        let u = BigInt::try_from_hex(s).unwrap();
        assert!(algorithm_eq(&u, &v));

        let s = "0x___bad_c0______________ffee__b_ad_cafe___";
        let u = BigInt::try_from_hex(s).unwrap();
        assert!(algorithm_eq(&u, &v));

        // Error states
        let s = "0x_";
        let u = BigInt::try_from_hex(s);
        assert!(matches!(u, Err(NoValidDigits)));

        let s = "1";
        let u = BigInt::try_from_hex(s);
        assert!(matches!(u, Err(MissingPrefix)));

        let s = "0xz";
        let u = BigInt::try_from_hex(s);
        assert!(matches!(u, Err(InvalidDigit)));

        let s = "0x_z___";
        let u = BigInt::try_from_hex(s);
        assert!(matches!(u, Err(InvalidDigit)));

        let s = "0x_ff_z";
        let u = BigInt::try_from_hex(s);
        assert!(matches!(u, Err(InvalidDigit)));

        let s = "";
        let u = BigInt::try_from_hex(s);
        assert!(matches!(u, Err(EmptyString)));

        let s = "0x";
        let u = BigInt::try_from_hex(s);
        assert!(matches!(u, Err(ZeroDigits)));
    }
}
