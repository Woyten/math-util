use num::Complex;
use num::Zero;
use rustfft::FFTplanner;
use std::sync::Mutex;

lazy_static! {
    static ref FORWARD_PLANNER: Mutex<FFTplanner<f32>> = Mutex::new(FFTplanner::new(false));
    static ref BACKWARD_PLANNER: Mutex<FFTplanner<f32>> = Mutex::new(FFTplanner::new(true));
}

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

pub fn transform(input: &mut [Complex<f32>], direction: TransformDirection) -> Vec<Complex<f32>> {
    let mut output = vec![Zero::zero(); input.len()];
    direction
        .get_planner()
        .lock()
        .unwrap()
        .plan_fft(input.len())
        .process(input, &mut output);
    output
}
