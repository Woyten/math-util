use nalgebra::DMatrix;
use num::Complex;
use num::Zero;
use rayon::prelude::*;
use rustfft::FFTplanner;
use std::borrow::BorrowMut;
use std::sync::Mutex;

lazy_static! {
    static ref FORWARD_PLANNER: Mutex<FFTplanner<f32>> = Mutex::new(FFTplanner::new(false));
    static ref BACKWARD_PLANNER: Mutex<FFTplanner<f32>> = Mutex::new(FFTplanner::new(true));
}

#[derive(Clone, Copy)]
pub enum TransformDirection {
    Forward,
    Backward,
}

impl TransformDirection {
    fn get_planner(&self) -> &'static Mutex<FFTplanner<f32>> {
        match *self {
            TransformDirection::Forward => &FORWARD_PLANNER,
            TransformDirection::Backward => &BACKWARD_PLANNER,
        }
    }
}

pub fn transform<I>(mut input: I, direction: TransformDirection) -> Vec<Complex<f32>>
where
    I: BorrowMut<[Complex<f32>]>,
{
    let input = input.borrow_mut();
    let mut output_buffer = vec![Zero::zero(); input.len()];
    transform_to(input, &mut output_buffer, direction);
    output_buffer
}

pub fn transform_to(input: &mut [Complex<f32>], output_buffer: &mut [Complex<f32>], direction: TransformDirection) {
    direction
        .get_planner()
        .lock()
        .unwrap()
        .plan_fft(input.len())
        .process(input, output_buffer);
}

pub fn transform_2d<I>(mut input: I, direction: TransformDirection) -> DMatrix<Complex<f32>>
where
    I: BorrowMut<DMatrix<Complex<f32>>>,
{
    let input = input.borrow_mut();
    let mut output_buffer = DMatrix::zeros(input.nrows(), input.ncols());
    transform_2d_to(input, &mut output_buffer, direction);
    output_buffer
}

pub fn transform_2d_to(input: &mut DMatrix<Complex<f32>>, output_buffer: &mut DMatrix<Complex<f32>>, direction: TransformDirection) {
    transform_cols(input, output_buffer, direction);
    let mut transposed_buffer = DMatrix::zeros(input.ncols(), input.nrows());
    transform_cols(&mut output_buffer.transpose(), &mut transposed_buffer, direction);
    transposed_buffer.transpose_to(output_buffer);
}

fn transform_cols(input: &mut DMatrix<Complex<f32>>, output_buffer: &mut DMatrix<Complex<f32>>, direction: TransformDirection) {
    let nrows = input.nrows();
    input
        .as_mut_slice()
        .par_chunks_mut(nrows)
        .zip(output_buffer.as_mut_slice().par_chunks_mut(nrows))
        .for_each(|(input, output_buffer)| transform_to(input, output_buffer, direction))
}
