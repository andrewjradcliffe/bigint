use std::fmt::Write;

pub mod parser;
pub mod u4;

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
        self.words.iter().all(|x| *x == 0)
        // This might be faster, depending on the internals of `all`
        // self.words.iter().fold(0x00, |acc, x| acc | *x) == 0x00
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
        // Literal implementation
        // let n = self.words.len();
        // let mut j: usize = 0;
        // let mut k: u8 = 0;
        // while j < n {
        //     let u_j = self.words[j];
        //     let w_j = u_j.wrapping_add(k).wrapping_add(u8::MAX);
        //     k = (w_j <= u_j) as u8; // (!(w_j > u_j)) as u8;
        //     self.words[j] = w_j;
        //     j += 1;
        // }
        // Rust-friendlier implementation
        let mut k: u8 = 0;
        for w_j in self.words.iter_mut() {
            if k == 1 {
                // The probability that the initial borrow will quickly
                // be extinguished is high, thus, this branch is worthwhile.
                break;
            } else {
                let w_j_prime = w_j.wrapping_add(k).wrapping_add(u8::MAX);
                k = (w_j_prime <= *w_j) as u8;
                *w_j = w_j_prime;
            }
        }
    }

    fn increment_with_carry(&mut self) -> u8 {
        // Literal implementation
        // let n = self.words.len();
        // let mut k: u8 = 1;
        // let mut j: usize = 0;
        // while j < n {
        //     if k == 0 {
        //         break;
        //     } else {
        //         let w_j = self.words[j];
        //         let w_j_prime = w_j.wrapping_add(k);
        //         k = (w_j_prime < w_j) as u8;
        //         self.words[j] = w_j_prime;
        //         j += 1;
        //     }
        // }
        // k
        // Rust-friendlier implementation
        let mut k: u8 = 1;
        for w_j in self.words.iter_mut() {
            if k == 0 {
                // The probability that the initial carry will quickly
                // be consumed is high, thus, this branch is worthwhile.
                break;
            } else {
                let w_j_prime = w_j.wrapping_add(k);
                k = (w_j_prime < *w_j) as u8;
                *w_j = w_j_prime;
            }
        }
        k
    }

    pub fn increment(&mut self) {
        let k = self.increment_with_carry();
        if k != 0 {
            self.words.push(k);
        }
    }

    pub fn modular_increment(&mut self) {
        self.increment_with_carry();
    }

    pub fn complement(&mut self) {
        self.words.iter_mut().for_each(|w| *w = !*w);
    }
    pub fn modular_negate(&mut self) {
        self.complement();
        self.modular_increment();
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
impl From<u8> for BigInt {
    fn from(bits: u8) -> Self {
        Self { words: vec![bits] }
    }
}
impl From<u16> for BigInt {
    fn from(bits: u16) -> Self {
        let w0 = bits as u8;
        let w1 = (bits >> 8) as u8;
        Self::from_rtol(&[w1, w0])
    }
}
impl From<u32> for BigInt {
    fn from(bits: u32) -> Self {
        let w0 = bits as u8;
        let w1 = (bits >> 8) as u8;
        let w2 = (bits >> 16) as u8;
        let w3 = (bits >> 24) as u8;
        Self::from_rtol(&[w3, w2, w1, w0])
    }
}
impl From<u64> for BigInt {
    fn from(bits: u64) -> Self {
        let w0 = bits as u8;
        let w1 = (bits >> 8) as u8;
        let w2 = (bits >> 16) as u8;
        let w3 = (bits >> 24) as u8;
        let w4 = (bits >> 32) as u8;
        let w5 = (bits >> 40) as u8;
        let w6 = (bits >> 48) as u8;
        let w7 = (bits >> 56) as u8;
        Self::from_rtol(&[w7, w6, w5, w4, w3, w2, w1, w0])
    }
}
impl From<u128> for BigInt {
    fn from(bits: u128) -> Self {
        let w0 = bits as u8;
        let w1 = (bits >> 8) as u8;
        let w2 = (bits >> 16) as u8;
        let w3 = (bits >> 24) as u8;
        let w4 = (bits >> 32) as u8;
        let w5 = (bits >> 40) as u8;
        let w6 = (bits >> 48) as u8;
        let w7 = (bits >> 56) as u8;
        let w8 = (bits >> 64) as u8;
        let w9 = (bits >> 72) as u8;
        let w10 = (bits >> 80) as u8;
        let w11 = (bits >> 88) as u8;
        let w12 = (bits >> 96) as u8;
        let w13 = (bits >> 104) as u8;
        let w14 = (bits >> 112) as u8;
        let w15 = (bits >> 120) as u8;
        Self::from_rtol(&[
            w15, w14, w13, w12, w11, w10, w9, w8, w7, w6, w5, w4, w3, w2, w1, w0,
        ])
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        algorithm_cmp(self, other)
    }
}
impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for BigInt {
    fn eq(&self, other: &Self) -> bool {
        algorithm_eq(self, other)
    }
}
impl Eq for BigInt {}

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

pub fn algorithm_a_varsize(u: &BigInt, v: &BigInt) -> (BigInt, u8) {
    let m = u.words.len();
    let n = v.words.len();
    if m < n {
        algorithm_a_varsize(v, u)
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
        (BigInt { words: w }, k)
    }
}

pub fn algorithm_a_sz(u: &BigInt, v: &BigInt) -> BigInt {
    let (mut w, k) = algorithm_a_varsize(u, v);
    if k != 0 {
        w.words.push(k)
    }
    w
}

pub fn algorithm_a_modular(u: &BigInt, v: &BigInt) -> BigInt {
    let (w, _) = algorithm_a_varsize(u, v);
    w
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
        // let w_j = u_j
        //     .wrapping_sub(v.words[j])
        //     .wrapping_add(k)
        //     .wrapping_add(u8::MAX);
        // k = (!(w_j > u_j)) as u8;
        let t = u_j.wrapping_sub(v.words[j]);
        let w_j = t.wrapping_add(k).wrapping_add(u8::MAX);
        k = (!(t > u_j)) as u8;
        w.push(w_j);
        j += 1;
    }
    BigInt { words: w }
}

pub fn algorithm_s_modular(u: &BigInt, v: &BigInt) -> (BigInt, u8) {
    // let n = u.words.len();
    // assert_eq!(n, v.words.len());

    // let mut w: Vec<u8> = Vec::with_capacity(n);
    // w.resize(n, 0);
    // let mut j: usize = 0;
    // let mut k: u8 = 1;
    // while j < n {
    //     let u_j = u.words[j];
    //     let w_j = u_j
    //         .wrapping_sub(v.words[j])
    //         .wrapping_add(k)
    //         .wrapping_add(u8::MAX);
    //     k = (!(w_j > u_j)) as u8;
    //     // CASE: if (u_j - v_j) + (b - 1) = 0 and k = 0
    //     // let t = u_j.wrapping_sub(v.words[j]).wrapping_add(u8::MAX);
    //     // if t == 0 && k == 0 {
    //     //     w[j] = t;
    //     //     k = (!(t > u_j)) as u8;
    //     // } else {
    //     //     let w_j = t.wrapping_add(k);
    //     //     k = (!(w_j > u_j)) as u8;
    //     //     w[j] = w_j;
    //     // }
    //     // let t = u_j.wrapping_sub(v.words[j]);
    //     // let w_j = t.wrapping_add(k).wrapping_add(u8::MAX);
    //     // k = ((!(t > u_j)) | (!(w_j > u_j))) as u8;
    //     w[j] = w_j;
    //     j += 1;
    // }
    // (BigInt { words: w }, k)
    let mut v = v.clone();
    v.modular_negate();
    algorithm_a_varsize(u, &v)
}
pub fn algorithm_s_modular_varsize(u: &BigInt, v: &BigInt) -> (BigInt, u8) {
    let mut v = v.clone();
    let m = u.words.len();
    let n = v.words.len();
    if m > n {
        for _ in 0..(m - n) {
            v.words.push(0);
        }
    }
    v.modular_negate();
    algorithm_a_varsize(u, &v)
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

pub fn algorithm_d(u: &BigInt, v: &BigInt) -> BigInt {
    let n = v.words.len();
    let m = u.words.len() - n;
    let mut q: Vec<u8> = Vec::with_capacity(m + 1);
    q.resize(m + 1, 0);
    let b: u16 = 1 << 8;
    let d = (b / (v.words[n - 1] as u16 + 1u16)) as u8;
    let mut u = algorithm_m(&u, &BigInt { words: vec![d] });
    if d == 1 {
        u.words.push(0);
    }
    // d is such that v * d will not increase the number of places
    let v = {
        let mut v = v.clone();
        let d = d as u16;
        let mut k: u16 = 0;
        let mut i: usize = 0;
        while i < n {
            let t = (v.words[i] as u16) * d + k;
            k = t >> 8;
            v.words[i] = (t & 0x00ff) as u8;
            i += 1;
        }
        v
    };
    let v_n_1 = v.words[n - 1] as u16;
    let v_n_2 = v.words[n - 2] as u16;
    let mut j: usize = m + 1;
    while j > 0 {
        j -= 1;
        let u_jn = u.words[j + n] as u16;
        let u_jn_1 = u.words[j + n - 1] as u16;
        let u_jn_2 = u.words[j + n - 2] as u16;
        let mut q_hat = (u_jn * b + u_jn_1) / v_n_1;
        let mut r_hat = (u_jn * b + u_jn_1) % v_n_1;
        while q_hat >= b || q_hat * v_n_2 > b * r_hat + u_jn_2 {
            q_hat -= 1;
            r_hat += v_n_1;
            if r_hat >= b {
                break;
            }
        }
        let u_t = &mut u.words[j..j + n + 1];
        assert_eq!(u_t.len(), n + 1);
        let mut i: usize = 0;
        let mut k: u8 = 1;
        let mut k_mul: u16 = 0;
        while i < n + 1 {
            let v_i = if i == n {
                k_mul as u8
            } else {
                let t = (v.words[i] as u16) * q_hat + k_mul;
                k_mul = t >> 8;
                (t & 0x00ff) as u8
            };
            let u_i = u_t[i];
            let w_i = u_i.wrapping_sub(v_i).wrapping_add(k).wrapping_add(u8::MAX);
            k = (!(w_i > u_i)) as u8;
            u_t[i] = w_i;
            i += 1;
        }
        q[j] = q_hat as u8;
        if k == 0 {
            // There is an extra borrow
            q[j] -= 1;
            let mut i: usize = 0;
            let mut k: u8 = 0;
            while i < n + 1 {
                let u_i = u_t[i];
                let v_i = if i == n { 0 } else { v.words[i] };
                let w_i = u_i.wrapping_add(v_i);
                let k_prime = w_i < u_i.max(v_i);
                let w_i_prime = w_i.wrapping_add(k);
                k = (k_prime | (w_i_prime < w_i)) as u8;
                u_t[i] = w_i_prime;
                i += 1;
            }
        }
    }
    BigInt { words: q }
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

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x00, 0x01]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0xff]);
        let w = algorithm_s(&u, &v);
        assert_eq!(w.words[0], 0x02);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x01, 0x01]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0xff]);
        let w = algorithm_s(&u, &v);
        assert_eq!(w.words[0], 0x02);
        assert_eq!(w.words[1], 0x01);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x02, 0x01]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0xff]);
        let w = algorithm_s(&u, &v);
        assert_eq!(w.words[0], 0x02);
        assert_eq!(w.words[1], 0x02);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x00, 0x00]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0x01]);
        let w = algorithm_s(&u, &v);
        assert_eq!(w.words[0], 0xff);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);
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
    fn division_works() {
        let u = BigInt::from_rtol(&[0xff, 0xff]);
        let v = BigInt::from_rtol(&[0x02, 0x00]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0x7f);

        let u = BigInt::from_rtol(&[0xff, 0xff]);
        let v = BigInt::from_rtol(&[0x01, 0x23]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0xe1);

        let u = BigInt::from_rtol(&[0xba, 0xde]);
        let v = BigInt::from_rtol(&[0x01, 0x23]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0xa4);

        let u = BigInt::from_rtol(&[0xff, 0xff]);
        let v = BigInt::from_rtol(&[0x01, 0x01]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0xff);

        let u = BigInt::from_rtol(&[0xc0, 0xca, 0xba, 0xde]);
        let v = BigInt::from_rtol(&[0x01, 0x23]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0x98);
        assert_eq!(w.words[1], 0x9a);
        assert_eq!(w.words[2], 0xa9);

        let u = BigInt::from_rtol(&[0xc0, 0xca, 0xba, 0xde]);
        let v = BigInt::from_rtol(&[0x12, 0x34]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0x55);
        assert_eq!(w.words[1], 0x97);
        assert_eq!(w.words[2], 0x0a);

        let u = BigInt::from_rtol(&[0x12, 0x34]);
        let v = BigInt::from_rtol(&[0xff, 0xff]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0x00);

        let u = BigInt::from_rtol(&[0x12, 0x34]);
        let v = BigInt::from_rtol(&[0xff, 0xff]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0x00);

        let u = BigInt::from_rtol(&[0x00, 0x00, 0x34]);
        let v = BigInt::from_rtol(&[0x01, 0x0f]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0x00);
        assert_eq!(w.words[1], 0x00);

        let u = BigInt::from_rtol(&[0x00, 0x00]);
        let v = BigInt::from_rtol(&[0x01, 0x00]);
        let w = algorithm_d(&u, &v);
        assert_eq!(w.words[0], 0x00);

        let u = BigInt::from_rtol(&[0xff, 0xff]);
        let v = BigInt::from_rtol(&[0xff, 0xff, 0xff, 0xff]);
        let p = algorithm_m(&u, &v);
        let q_u = algorithm_d(&p, &u);
        assert!(algorithm_eq(&q_u, &v));
        let q_v = algorithm_d(&p, &v);
        assert!(algorithm_eq(&q_v, &u));
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

        let u = BigInt::from(0x1230_u16);
        let v = BigInt::from(0x1201_u16);
        assert!(v < u);
        assert!(v <= u);
        assert!(!(u < v));
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

        let u = BigInt::from(0x1230_u16);
        let v = BigInt::from(0x1201_u16);
        assert!(u > v);
        assert!(u >= v);
        assert!(!(u < v));
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

        let u = BigInt::from(0x1234_u16);
        let v = u.clone();
        assert_eq!(u, v);

        let v = BigInt::from(0x00_u8);
        assert_ne!(u, v);
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

        let mut u = BigInt::from_rtol(&[0x03]);
        u.decrement();
        let v = BigInt::from_rtol(&[0x02]);
        assert!(algorithm_eq(&u, &v));

        let mut u = BigInt::from_rtol(&[0x01, 0x00]);
        u.decrement();
        let v = BigInt::from_rtol(&[0x00, 0xff]);
        assert!(algorithm_eq(&u, &v));

        let mut u = BigInt::from_rtol(&[0x01, 0xff, 0x00]);
        u.decrement();
        let v = BigInt::from_rtol(&[0x01, 0xfe, 0xff]);
        assert!(algorithm_eq(&u, &v));

        let mut u = BigInt::from_rtol(&[0x01, 0xff, 0x00, 0x00, 0x00]);
        u.decrement();
        let v = BigInt::from_rtol(&[0x01, 0xfe, 0xff, 0xff, 0xff]);
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

        let mut u = BigInt::from_rtol(&[0x01, 0xfe]);
        u.increment();
        let v = BigInt::from_rtol(&[0x01, 0xff]);
        assert!(algorithm_eq(&u, &v));

        let mut u = BigInt::from_rtol(&[0x01, 0xf0]);
        for _ in 0..16 {
            u.increment();
        }
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
    fn modular_increment_works() {
        let mut u = BigInt::from_rtol(&[0xff]);
        u.modular_increment();
        assert_eq!(u.words.len(), 1);
        assert!(algorithm_eq(&u, &u));

        let mut u = BigInt::from_rtol(&[0xff, 0xff, 0xff]);
        u.modular_increment();
        assert_eq!(u.words.len(), 3);
        assert!(algorithm_eq(&u, &u));
    }

    #[test]
    fn complement_works() {
        let mut u = BigInt::from_rtol(&[0xff]);
        u.complement();
        assert_eq!(u.words.len(), 1);
        assert!(algorithm_eq(&u, &BigInt::from_rtol(&[0x00])));

        let mut u = BigInt::from_rtol(&[0xf0, 0x0f, 0xee]);
        u.complement();
        assert_eq!(u.words.len(), 3);
        assert!(algorithm_eq(&u, &BigInt::from_rtol(&[0x0f, 0xf0, 0x11])));
    }

    #[test]
    fn modular_negate_works() {
        let mut u = BigInt::from_rtol(&[0x7f]);
        u.modular_negate();
        assert_eq!(u.words.len(), 1);
        assert!(algorithm_eq(&u, &BigInt::from_rtol(&[0x81])));

        let mut u = BigInt::from_rtol(&[0x00, 0x07]);
        u.modular_negate();
        assert_eq!(u.words.len(), 2);
        assert!(algorithm_eq(&u, &BigInt::from_rtol(&[0xff, 0xf9])));

        // let mut u = BigInt::from_rtol(&[0xff]);
        // u.negate();
        // assert!(algorithm_eq(&u, &BigInt::from_rtol(&[0x81])));

        // Negation to implement subtraction
        let u = BigInt::from_rtol(&[0x01]);
        let mut v = BigInt::from_rtol(&[0x07]);
        v.modular_negate();
        let w = algorithm_a(&u, &v);
        assert!(algorithm_eq(&w, &BigInt::from_rtol(&[0xfa])));

        let u = BigInt::from_rtol(&[0x00, 0x01]);
        let mut v = BigInt::from_rtol(&[0x00, 0x07]);
        v.modular_negate();
        let w = algorithm_a(&u, &v);
        assert!(algorithm_eq(&w, &BigInt::from_rtol(&[0xff, 0xfa])));

        let u = BigInt::from_rtol(&[0x01]);
        let mut v = BigInt::from_rtol(&[0x00, 0x07]);
        v.modular_negate();
        let w = algorithm_a_sz(&u, &v);
        assert!(algorithm_eq(&w, &BigInt::from_rtol(&[0xff, 0xfa])));
    }

    #[test]
    fn sub_using_modular_negation_works() {
        let u = BigInt::from_rtol(&[0xba, 0xdc, 0x0f]);
        let v = BigInt::from_rtol(&[0x00, 0xaa, 0xbb]);

        let lhs = algorithm_s(&u, &v);

        let mut n_v = v.clone();
        n_v.modular_negate();

        let rhs = algorithm_a_modular(&u, &n_v);
        assert!(algorithm_eq(&lhs, &rhs));

        let u = BigInt::from_rtol(&[0x01, 0x00]);
        let mut v = BigInt::from_rtol(&[0x02, 0x00]);
        v.modular_negate();

        let w = algorithm_a_modular(&u, &v);
        assert!(algorithm_eq(&w, &BigInt::from_rtol(&[0xff, 0x00])));

        let u = BigInt::from_rtol(&[0x01, 0x00]);
        let mut v = BigInt::from_rtol(&[0x80, 0x00, 0x00, 0x00]);
        v.modular_negate();

        let w = algorithm_a_modular(&u, &v);
        assert!(algorithm_eq(
            &w,
            &BigInt::from_rtol(&[0x80, 0x00, 0x01, 0x00])
        ));
    }

    #[test]
    fn modular_sub_works() {
        let u = BigInt::from_rtol(&[0x00, 0x00, 0x00]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0x01]);
        let (mut w, k) = algorithm_s_modular(&u, &v);
        assert_eq!(w.words[0], 0xff);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(w.words[2], 0xff);
        assert_eq!(k, 0);

        w.modular_negate();
        assert_eq!(w.words[0], 0x01);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(w.words[2], 0x00);

        let u = BigInt::from_rtol(&[0x00, 0x05]);
        let v = BigInt::from_rtol(&[0x00, 0x0f]);
        let (mut w, k) = algorithm_s_modular(&u, &v);
        assert_eq!(w.words[0], 0xf6);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(k, 0);

        w.modular_negate();
        assert_eq!(w.words[0], 0x0a);
        assert_eq!(w.words[1], 0x00);

        let u = BigInt::from_rtol(&[0x00, 0x05]);
        let v = BigInt::from_rtol(&[0x01, 0x00]);
        let (mut w, k) = algorithm_s_modular(&u, &v);
        assert_eq!(w.words[0], 0x05);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(k, 0);

        w.modular_negate();
        assert_eq!(w.words[0], 0xfb);
        assert_eq!(w.words[1], 0x00);

        let u = BigInt::from_rtol(&[0x01, 0x00]);
        let v = BigInt::from_rtol(&[0x00, 0x05]);
        let (w, k) = algorithm_s_modular(&u, &v);
        assert_eq!(w.words[0], 0xfb);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(k, 1);

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x01, 0x00]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0xff]);
        // v.modular_negate();
        // let w = algorithm_a_modular(&u, &v);
        let (w, _) = algorithm_s_modular(&u, &v);
        assert_eq!(w.words[0], 0x01);
        assert_eq!(w.words[1], 0x01);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x00, 0x00]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0xff]);
        let (w, _) = algorithm_s_modular(&u, &v);
        assert_eq!(w.words[0], 0x01);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x00, 0x01]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0xff, 0xff]);
        let (w, _) = algorithm_s_modular(&u, &v);
        assert_eq!(w.words[0], 0x02);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);
    }

    #[test]
    fn modular_sub_varsize_works() {
        // m < n
        let u = BigInt::from_rtol(&[0x00]);
        let v = BigInt::from_rtol(&[0x00, 0x00, 0x01]);
        let (w, k) = algorithm_s_modular_varsize(&u, &v);
        assert_eq!(w.words[0], 0xff);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(w.words[2], 0xff);
        assert_eq!(k, 0);

        let u = BigInt::from_rtol(&[0x0f]);
        let v = BigInt::from_rtol(&[0x01, 0x01]);
        let (mut w, k) = algorithm_s_modular_varsize(&u, &v);
        assert_eq!(w.words[0], 0x0e);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(k, 0);

        w.modular_negate();
        assert_eq!(w.words[0], 242); // 0xf2
        assert_eq!(w.words[1], 0x00);

        // m = n
        let u = BigInt::from_rtol(&[0x07]);
        let v = BigInt::from_rtol(&[0x09]);
        let (mut w, k) = algorithm_s_modular_varsize(&u, &v);
        assert_eq!(w.words[0], 0xfe);
        assert_eq!(k, 0);

        w.modular_negate();
        assert_eq!(w.words[0], 0x02);

        // m > n
        let u = BigInt::from_rtol(&[0x00, 0x00, 0x01]);
        let v = BigInt::from_rtol(&[0x02]);
        let (mut w, k) = algorithm_s_modular_varsize(&u, &v);
        assert_eq!(w.words[0], 0xff);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(w.words[2], 0xff);
        assert_eq!(k, 0);

        w.modular_negate();
        assert_eq!(w.words[0], 0x01);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(w.words[2], 0x00);

        let u = BigInt::from_rtol(&[0x01, 0x01, 0x00, 0x01]);
        let v = BigInt::from_rtol(&[0xff, 0xff]);
        let (w, k) = algorithm_s_modular_varsize(&u, &v);
        assert_eq!(w.words[0], 0x02);
        assert_eq!(w.words[1], 0x00);
        assert_eq!(w.words[2], 0x00);
        assert_eq!(w.words[3], 0x01);
        assert_eq!(k, 1);

        let u = BigInt::from_rtol(&[0xff, 0xff]);
        let v = BigInt::from_rtol(&[0x01, 0x01, 0x00, 0x01]);
        let (w, k) = algorithm_s_modular_varsize(&u, &v);
        assert_eq!(w.words[0], 0xfe);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(w.words[2], 0xff);
        assert_eq!(w.words[3], 0xfe);
        assert_eq!(k, 0);

        let u = BigInt::from_rtol(&[0xff, 0xff]);
        let v = BigInt::from_rtol(&[0x01, 0x00, 0x00]);
        let (w, k) = algorithm_s_modular_varsize(&u, &v);
        assert_eq!(w.words[0], 0xff);
        assert_eq!(w.words[1], 0xff);
        assert_eq!(w.words[2], 0xff);
        assert_eq!(k, 0);
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

#[derive(Debug, Clone)]
pub struct SignMag {
    sign: bool,
    mag: BigInt,
}
impl SignMag {
    pub fn is_negative(&self) -> bool {
        self.sign
    }
    pub fn is_positive(&self) -> bool {
        !self.sign
    }
    pub fn is_zero(&self) -> bool {
        self.mag.is_zero()
    }
    pub fn is_even(&self) -> bool {
        self.mag.is_even()
    }
    pub fn is_odd(&self) -> bool {
        self.mag.is_odd()
    }
}

use std::ops::Add;
impl Add for &SignMag {
    type Output = SignMag;
    fn add(self, other: Self) -> SignMag {
        let lhs = self;
        let rhs = other;
        if lhs.sign & rhs.sign {
            SignMag {
                sign: true,
                mag: algorithm_a_sz(&lhs.mag, &rhs.mag),
            }
        } else if lhs.sign & !rhs.sign {
            let (mut mag, k) = algorithm_s_modular_varsize(&rhs.mag, &lhs.mag);
            let sign = if k == 0 {
                mag.modular_negate();
                if mag.is_zero() {
                    false
                } else {
                    true
                }
            } else {
                false
            };
            SignMag { sign, mag }
        } else if !lhs.sign & rhs.sign {
            let (mut mag, k) = algorithm_s_modular_varsize(&lhs.mag, &rhs.mag);
            let sign = if k == 0 {
                mag.modular_negate();
                if mag.is_zero() {
                    false
                } else {
                    true
                }
            } else {
                false
            };
            SignMag { sign, mag }
        } else {
            SignMag {
                sign: false,
                mag: algorithm_a_sz(&lhs.mag, &rhs.mag),
            }
        }
    }
}
use std::ops::Sub;
impl Sub for &SignMag {
    type Output = SignMag;
    fn sub(self, other: Self) -> SignMag {
        let lhs = self;
        let rhs = other;
        if lhs.sign & rhs.sign {
            let (mut mag, k) = algorithm_s_modular_varsize(&rhs.mag, &lhs.mag);
            let sign = if k == 0 {
                mag.modular_negate();
                if mag.is_zero() {
                    false
                } else {
                    true
                }
            } else {
                false
            };
            SignMag { sign, mag }
        } else if !lhs.sign & rhs.sign {
            let mag = algorithm_a_sz(&lhs.mag, &rhs.mag);
            SignMag { sign: false, mag }
        } else if lhs.sign & !rhs.sign {
            let mag = algorithm_a_sz(&lhs.mag, &rhs.mag);
            SignMag { sign: true, mag }
        } else {
            let (mut mag, k) = algorithm_s_modular_varsize(&lhs.mag, &rhs.mag);
            let sign = if k == 0 {
                mag.modular_negate();
                if mag.is_zero() {
                    false
                } else {
                    true
                }
            } else {
                false
            };
            SignMag { sign, mag }
        }
    }
}

impl From<BigInt> for SignMag {
    fn from(mag: BigInt) -> Self {
        Self { sign: false, mag }
    }
}
macro_rules! impl_from_uN {
    { $uN:ident } => {
        impl From<$uN> for SignMag {
            fn from(bits: $uN) -> Self {
                Self {
                    sign: false,
                    mag: BigInt::from(bits),
                }
            }
        }
    }
}
impl_from_uN! { u8 }
impl_from_uN! { u16 }
impl_from_uN! { u32 }
impl_from_uN! { u64 }
impl_from_uN! { u128 }

use std::ops::Mul;
impl Mul for &SignMag {
    type Output = SignMag;
    fn mul(self, other: Self) -> SignMag {
        let mag = algorithm_m(&self.mag, &other.mag);
        let sign = self.sign ^ other.sign;
        SignMag { sign, mag }
    }
}
use std::ops::Div;
impl Div for &SignMag {
    type Output = SignMag;
    fn div(self, other: Self) -> SignMag {
        let mag = algorithm_d(&self.mag, &other.mag);
        let sign = self.sign ^ other.sign;
        SignMag { sign, mag }
    }
}
use std::ops::Neg;
impl Neg for &SignMag {
    type Output = SignMag;
    fn neg(self) -> SignMag {
        let mag = self.mag.clone();
        if mag.is_zero() {
            SignMag { sign: false, mag }
        } else {
            SignMag {
                sign: !(self.sign),
                mag,
            }
        }
    }
}
use std::cmp::Eq;
use std::cmp::Ord;

impl Ord for SignMag {
    fn cmp(&self, other: &Self) -> Ordering {
        if !self.sign & !other.sign {
            algorithm_cmp(&self.mag, &other.mag)
            // self.mag.cmp(&other.mag)
        } else if !self.sign & other.sign {
            Ordering::Greater
        } else if self.sign & !other.sign {
            Ordering::Less
        } else {
            match algorithm_cmp(&self.mag, &other.mag) {
                Ordering::Less => Ordering::Greater,
                Ordering::Greater => Ordering::Less,
                Ordering::Equal => Ordering::Equal,
            }
        }
    }
}
impl PartialOrd for SignMag {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SignMag {
    fn eq(&self, other: &Self) -> bool {
        if self.sign ^ other.sign {
            false
        } else {
            algorithm_eq(&self.mag, &other.mag)
        }
    }
}
impl Eq for SignMag {}

#[cfg(test)]
mod signmag_tests {
    use super::*;

    #[test]
    fn neg_works() {
        let u = SignMag::from(0xff_u8);
        let v = -&u;
        assert_eq!(
            v,
            SignMag {
                sign: true,
                mag: BigInt::from(0xff_u8)
            }
        );
    }

    #[test]
    fn add_works() {
        let u = SignMag::from(0x01ff_u16);
        let v = SignMag::from(0x01_u8);
        let w = &u + &v;
        assert_eq!(w.mag.words[0], 0x00);
        assert_eq!(w.mag.words[1], 0x02);

        let a = 0xf1_u8;
        let b = 0x0f_u8;
        let u = SignMag::from(a);
        let v = SignMag::from(b);
        assert_eq!(&u + &v, SignMag::from((a as u16) + (b as u16)));

        let u = SignMag::from(0x01_u8);
        let v = SignMag::from(0x02_u8);
        assert_eq!(u, &v + &(-&u));
    }

    #[test]
    fn sub_works() {
        let u = SignMag::from(0x01ff_u16);
        let v = SignMag::from(0x01_u8);
        let w = &u - &v;
        assert_eq!(w.mag.words[0], 0xfe);
        assert_eq!(w.mag.words[1], 0x01);

        let u = SignMag::from(0x01ff_u16);
        let v = u.clone();
        let w = &u - &v;
        assert_eq!(w.mag.words[0], 0x00);
        assert_eq!(w.mag.words[1], 0x00);

        let v = SignMag::from(0x03ff_u16);

        let w = &u - &v;
        assert_eq!(w.mag.words[0], 0x00);
        assert_eq!(w.mag.words[1], 0x02);
        assert_eq!(w.sign, true);
    }

    #[test]
    fn mul_works() {
        let u = SignMag::from(BigInt::from_rtol(&[0xff]));
        let v = SignMag::from(BigInt::from_rtol(&[0xff]));
        let w = &u * &v;
        assert_eq!(w.mag.words[0], 0x01);
        assert_eq!(w.mag.words[1], 0xfe);
    }

    // #[test]
    // fn div_works() {
    //     let u = SignMag::from(BigInt::from_rtol(&[0xff]));
    //     let v = SignMag::from(BigInt::from_rtol(&[0xff, 0x00]));
    //     let w = &u * &v;
    //     let q = &w / &v;
    //     assert_eq!(q.mag.words[0], 0x00);
    //     assert_eq!(q.mag.words[1], 0x00);
    // }
}
