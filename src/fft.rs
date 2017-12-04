use matrix;
use nalgebra::DMatrix;
use num::Complex;
use num::Zero;
use rustfft::FFTplanner;
use std::cell::RefCell;
use std::sync::Mutex;
use thread_local::CachedThreadLocal;

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

pub fn transform(input: &mut [Complex<f32>], direction: TransformDirection) {
    let len = input.len();
    transform_using_buffer(input, direction, &mut vec![Zero::zero(); len]);
}

pub fn transform_using_buffer(input: &mut [Complex<f32>], direction: TransformDirection, output_buffer: &mut Vec<Complex<f32>>) {
    direction
        .get_planner()
        .lock()
        .unwrap()
        .plan_fft(input.len())
        .process(input, output_buffer);

    input.copy_from_slice(&output_buffer);
}

type CachedBuffer = CachedThreadLocal<RefCell<Vec<Complex<f32>>>>;

pub fn transform_2d(input: &mut DMatrix<Complex<f32>>, direction: TransformDirection) {
    transform_cols(input, direction);
    let mut transposed = input.transpose();
    transform_cols(&mut transposed, direction);
    transposed.transpose_to(input);
}

fn transform_cols(input: &mut DMatrix<Complex<f32>>, direction: TransformDirection) {
    let nrows = input.nrows();
    let output_buffer = CachedBuffer::new();
    matrix::map_cols_in_place(input, |col| {
        let output_buffer = output_buffer.get_or(|| Box::new(RefCell::new(vec![Zero::zero(); nrows])));
        transform_using_buffer(col, direction, &mut output_buffer.borrow_mut());
    });
}
