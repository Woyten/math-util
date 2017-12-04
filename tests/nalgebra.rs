extern crate nalgebra;

use nalgebra::Dynamic;
use nalgebra::MatrixMN;

#[test]
fn store_matrix_components_in_column_major_order() {
    let input_components = [1, 2, 3, 4];

    let dim = Dynamic::new(2);
    assert_eq!(MatrixMN::from_column_slice_generic(dim, dim, &input_components).as_slice(), &[1, 2, 3, 4]);
    assert_eq!(MatrixMN::from_row_slice_generic(dim, dim, &input_components).as_slice(), &[1, 3, 2, 4]);
}
