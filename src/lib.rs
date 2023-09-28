#[derive(Debug, Clone)]
pub struct BigInt {
    words: Vec<u8>,
}
impl BigInt {
    pub fn from_rtol(rtol_words: &[u8]) -> Self {
        Self {
            words: rtol_words.iter().cloned().rev().collect(),
        }
    }

    // // Not correct!
    // pub fn print(&self) {
    //     let mut s = String::new();
    //     for w in self.words.iter() {
    //         let mut lhs = format!("{:03}", w);
    //         lhs.push_str(&s);
    //         s = lhs;
    //     }
    //     println!("{}", s);
    // }
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
    w.push(k);
    BigInt { words: w }
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

fn lt_same_size(u: &[u8], v: &[u8]) -> bool {
    let m = u.len();
    assert_eq!(m, v.len());
    let mut i: usize = m;
    let mut state = false;
    while i > 0 {
        i -= 1;
        if u[i] == v[i] {
            continue;
        } else if u[i] > v[i] {
            break;
        } else {
            state = true;
            break;
        }
    }
    state
}

pub fn algorithm_lt(u: &BigInt, v: &BigInt) -> bool {
    let m = u.words.len();
    let n = v.words.len();
    if m < n {
        let mut i: usize = n;
        while i > m {
            i -= 1;
            if v.words[i] != 0 {
                return true;
            }
        }
        lt_same_size(&u.words[0..i], &v.words[0..i])
    } else if m == n {
        lt_same_size(&u.words[..], &v.words[..])
    } else {
        // m > n
        let mut i: usize = m;
        while i > n {
            i -= 1;
            if u.words[i] != 0 {
                return false;
            }
        }
        lt_same_size(&u.words[0..i], &v.words[0..i])
    }
}

pub fn algorithm_ge(u: &BigInt, v: &BigInt) -> bool {
    !algorithm_lt(u, v)
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
    fn lt_works() {
        let u = BigInt::from_rtol(&[0x01, 0x00]);
        let v = BigInt::from_rtol(&[0x02, 0x01]);
        assert!(algorithm_lt(&u, &v));

        let w = BigInt::from_rtol(&[0x00, 0x00, 0x02, 0x01]);
        assert!(algorithm_lt(&u, &w));

        let x = BigInt::from_rtol(&[0x00, 0x01, 0x02, 0x01]);
        assert!(!algorithm_lt(&x, &v));
    }
}
