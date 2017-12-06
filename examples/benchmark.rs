extern crate math_util;
extern crate nalgebra;
extern crate num;
extern crate rand;

use math_util::fft;
use math_util::fft::TransformDirection;
use nalgebra::DMatrix;
use nalgebra::Dynamic;
use num::Complex;
use rand::Rng;
use std::time::Instant;

fn main() {
    for _ in 0..10 {
        bench_fft_2d();
    }
}

fn bench_fft_2d() {
    let mut rng = rand::thread_rng();

    let size = Dynamic::new(3000);
    let large_matrix = DMatrix::from_fn_generic(size, size, |_, _| Complex::new(rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0)));

    let start = Instant::now();
    fft::transform_2d(large_matrix, TransformDirection::Forward);
    let elapsed = start.elapsed();

    println!("2D FFT took {}ms", elapsed.as_secs() * 1000 + elapsed.subsec_nanos() as u64 / 1000000);
}
