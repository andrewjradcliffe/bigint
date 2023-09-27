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

// use std::ops::Add;
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

pub fn algorithm_ge(u: &BigInt, v: &BigInt) -> bool {
    let n = u.words.len();
    assert_eq!(n, v.words.len());
    u.words[n - 1] >= v.words[n - 1]
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
}
