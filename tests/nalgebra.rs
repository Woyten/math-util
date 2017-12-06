extern crate nalgebra;

use nalgebra::DMatrix;
use nalgebra::Dynamic;

#[test]
fn store_matrix_components_in_column_major_order() {
    let input_components = [1, 2, 3, 4];

    let dim = Dynamic::new(2);
    assert_eq!(DMatrix::from_column_slice_generic(dim, dim, &input_components).as_slice(), &[1, 2, 3, 4]);
    assert_eq!(DMatrix::from_row_slice_generic(dim, dim, &input_components).as_slice(), &[1, 3, 2, 4]);
}
