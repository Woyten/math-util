use nalgebra::DMatrix;

#[test]
fn store_matrix_components_in_column_major_order() {
    let input_components = [1, 2, 3, 4];

    assert_eq!(
        DMatrix::from_column_slice(2, 2, &input_components).as_slice(),
        &[1, 2, 3, 4]
    );
    assert_eq!(
        DMatrix::from_row_slice(2, 2, &input_components).as_slice(),
        &[1, 3, 2, 4]
    );
}
