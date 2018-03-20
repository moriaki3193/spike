mod linalg;

#[cfg(test)]
mod tests {
    use linalg::{gen_rand_val, gen_rand_vec, gen_rand_mat, inner_prod, transpose};

    #[test]
    fn linalg() {
        let vec = gen_rand_vec(10usize, 0f64, 0.01f64);
        let mat = gen_rand_mat(5usize, 3usize, 0f64, 0.01f64);
        let transposed_mat = transpose(&mat);

        assert_eq!(vec.len(), 10usize);

        let u: Vec<f64> = vec![1.0, 2.0, 3.0];
        let v: Vec<f64> = vec![4.0, 5.0, 6.0];
        assert_eq!(inner_prod(&u, &v), 32f64);

        assert_eq!(mat.len(), transposed_mat[0].len());
        assert_eq!(mat[0].len(), transposed_mat.len());
    }
}
