extern crate rand;
use self::rand::{thread_rng, Rng};
use std::f64::consts::PI;

pub fn gen_rand_val(mu: f64, sigma: f64) -> f64 {
    /* Box-Muller method */
    let mut rng = thread_rng();
    let u1: f64 = rng.gen_range(0f64, 1f64);
    let u2: f64 = rng.gen_range(0f64, 1f64);
    let a = (- 2f64 * u1.ln()).sqrt();
    let b = (2f64 * PI * u2).cos();
    mu + sigma * a * b
}

pub fn gen_rand_vec(size: usize, mu: f64, sigma: f64) -> Vec<f64> {
    let mut vec = vec![];
    for _ in 0..size {
        let elem = gen_rand_val(mu, sigma);
        vec.push(elem);
    }
    vec
}

pub fn gen_rand_mat(nrow: usize, ncol: usize, mu: f64, sigma: f64) -> Vec<Vec<f64>> {
    let mut mat = vec![];
    for _i in 0..nrow {
        let mut vec = vec![];
        for _j in 0..ncol {
            let elem = gen_rand_val(mu, sigma);
            vec.push(elem);
        }
        mat.push(vec);
    }
    mat
}

pub fn inner_prod(u: &Vec<f64>, v: &Vec<f64>) -> f64 {
    // TODO use guard
    if u.len() != v.len() {
        panic!("u.len() must be same as v.len()");
    }
    let mut res: f64 = 0.0;
    for (i, v_i) in v.iter().enumerate() {
        res += v_i * u[i];
    }
    res
}

pub fn transpose(v: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let nrows = v.len();
    let ncols = v[0].len();
    let mut transposed = vec![];
    for i in 0..ncols {
        let mut rowvec = vec![];
        for j in 0..nrows {
            rowvec.push(v[j][i]);
        }
        transposed.push(rowvec);
    }
    transposed
}
