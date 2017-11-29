extern crate math_util;
extern crate num;

use math_util::fft;
use math_util::fft::TransformDirection;
use num::Complex;

#[test]
fn sanity_test() {
    let mut input = vec![
        Complex::new(-1.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    let mut transformed = fft::transform(&mut input, TransformDirection::Forward);
    let output = fft::transform(&mut transformed, TransformDirection::Backward);

    let expected_output = vec![
        Complex::new(-3.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    assert_eq!(output, expected_output);
}
