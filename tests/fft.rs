extern crate math_util;
extern crate nalgebra;
extern crate num;

use math_util::fft;
use math_util::fft::TransformDirection;
use nalgebra::Dynamic;
use nalgebra::MatrixMN;
use num::Complex;

#[test]
fn sanity_test() {
    let mut value_to_transform = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(-1.0, 0.0),
    ];

    fft::transform(&mut value_to_transform, TransformDirection::Forward);
    fft::transform(&mut value_to_transform, TransformDirection::Backward);

    let expected_output = vec![
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 3.0),
        Complex::new(-3.0, 0.0),
    ];

    assert_eq!(value_to_transform, expected_output);
}

#[test]
fn sanity_test_2d() {
    let components = [
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, -1.0),
    ];
    let mut value_to_transform = MatrixMN::from_column_slice_generic(Dynamic::new(3), Dynamic::new(2), &components);

    fft::transform_2d(&mut value_to_transform, TransformDirection::Forward);
    fft::transform_2d(&mut value_to_transform, TransformDirection::Backward);

    let components = [
        Complex::new(6.0, 0.0),
        Complex::new(0.0, 6.0),
        Complex::new(-6.0, 0.0),
        Complex::new(0.0, 6.0),
        Complex::new(6.0, 0.0),
        Complex::new(0.0, -6.0),
    ];
    let expected_output = MatrixMN::from_column_slice_generic(Dynamic::new(3), Dynamic::new(2), &components);

    assert_eq!(value_to_transform, expected_output);
}
