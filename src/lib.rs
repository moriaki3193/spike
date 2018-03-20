mod linalg;
mod fm;

#[cfg(test)]
mod tests {
    use linalg::{gen_rand_val, gen_rand_vec, gen_rand_mat, inner_prod, transpose};
    use fm::{FM};

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

    #[test]
    fn fm_learning() {
        let features = vec![
            //       Users    |         Movies      |     Movie Ratings   |   Time |   Last Movie Rated   |
            //    A    B    C |   TI   NH   SW   ST |   TI   NH   SW   ST |        |   TI   NH   SW   ST  |
            vec![1.0, 0.0, 0.0,   1.0, 0.0, 0.0, 0.0,   0.3, 0.3, 0.3, 0.0,   13.0,    0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0,   0.3, 0.3, 0.3, 0.0,   14.0,    1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0,   0.0, 0.0, 1.0, 0.0,   0.3, 0.3, 0.3, 0.0,   16.0,    0.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0,   0.0, 0.0, 1.0, 0.0,   0.0, 0.0, 0.5, 0.5,   5.0,     0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0,   0.0, 0.0, 0.0, 1.0,   0.0, 0.0, 0.5, 0.5,   8.0,     0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.1,   1.0, 0.0, 0.0, 0.0,   0.5, 0.0, 0.5, 0.0,   9.0,     0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.1,   0.0, 0.0, 1.0, 0.0,   0.5, 0.0, 0.5, 0.0,   12.0,    1.0, 0.0, 0.0, 0.0],
        ];
        let target = vec![5.0, 3.0, 1.0, 4.0, 5.0, 1.0, 5.0];

        let ndim: usize = features[0].len();
        let k: usize = 8;
        let mut model = FM::new(&k, &ndim);

        println!();
        println!("[FM model explanation]");
        println!("----------------------");
        println!("k of model is    : {}", &model.k);
        println!("nrow of model is : {}", &model.v.len());
        println!("ncol of model is : {}", &model.v[0].len());
        println!("----------------------");
        println!();

        /* Train & Test split */
        let mut x_train = vec![];
        let mut x_test = vec![];
        let mut y_train = vec![];
        let mut y_test = vec![];
        for i in 0..5 {
            x_train.push(&features[i]);
            y_train.push(target[i]);
        }
        for i in 5..7 {
            x_test.push(&features[i]);
            y_test.push(target[i]);
        }

        /* Training */
        let epochs: i32 = 5;
        model.fit(&x_train, &y_train, epochs);
        println!("# of iteration: {}", epochs);

        /* Test */
        let mut t_pred = vec![];
        for i in 0..2 {
            let y_pred = model.predict_one(&x_test[i]);
            t_pred.push(y_pred);
            println!("----- Test No: {} -----", i + 1);
            println!("y observed : {}", y_test[i]);
            println!("y predicted: {}", y_pred);
        }

        let t_pred_bulk = model.predict(&x_test);
        assert_eq!(t_pred[0], t_pred_bulk[0]);
        assert_eq!(t_pred[1], t_pred_bulk[1]);
    }
}
