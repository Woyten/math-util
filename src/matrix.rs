use nalgebra::DMatrix;
use num::Complex;
use rayon::prelude::*;

pub fn map_cols_in_place<F>(input: &mut DMatrix<Complex<f32>>, mapper: F)
where
    F: Fn(&mut [Complex<f32>]) + Send + Sync,
{
    let nrows = input.nrows();
    input.as_mut_slice().par_chunks_mut(nrows).for_each(mapper);
}
