#[derive(Debug)]
pub enum ParseBigIntError {
    MissingPrefix,
    InvalidDigit,
    ZeroDigits,
    NoValidDigits,
    EmptyString,
}

pub fn is_hex_digit(c: char) -> bool {
    match c {
        '0'..='9' | 'a'..='f' | 'A'..='F' => true,
        _ => false,
    }
}

pub fn char_to_hex(c: char) -> u8 {
    match c {
        '0' => 0x00,
        '1' => 0x01,
        '2' => 0x02,
        '3' => 0x03,
        '4' => 0x04,
        '5' => 0x05,
        '6' => 0x06,
        '7' => 0x07,
        '8' => 0x08,
        '9' => 0x09,
        'a' | 'A' => 0x0a,
        'b' | 'B' => 0x0b,
        'c' | 'C' => 0x0c,
        'd' | 'D' => 0x0d,
        'e' | 'E' => 0x0e,
        'f' | 'F' => 0x0f,
        _ => 0xff,
    }
}
