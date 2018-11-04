use math_util::fft;
use math_util::fft::TransformDirection;
use nalgebra::DMatrix;
use rustfft::num_complex::Complex;

#[test]
fn sanity_test() {
    let input = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(-1.0, 0.0),
    ];

    let transformed = fft::transform(input, TransformDirection::Forward);
    let output = fft::transform(transformed, TransformDirection::Backward);

    let expected_output = vec![
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 3.0),
        Complex::new(-3.0, 0.0),
    ];

    assert_eq!(output, expected_output);
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
    let input = DMatrix::from_column_slice(3, 2, &components);

    let transformed = fft::transform_2d(input, TransformDirection::Forward);
    let output = fft::transform_2d(transformed, TransformDirection::Backward);

    let components = [
        Complex::new(6.0, 0.0),
        Complex::new(0.0, 6.0),
        Complex::new(-6.0, 0.0),
        Complex::new(0.0, 6.0),
        Complex::new(6.0, 0.0),
        Complex::new(0.0, -6.0),
    ];
    let expected_output = DMatrix::from_column_slice(3, 2, &components);

    assert_eq!(output, expected_output);
}
