extern crate num;

use linalg::{gen_rand_val, gen_rand_vec, gen_rand_mat, inner_prod, transpose};
use self::num::pow;

pub struct FM {
    pub k: usize,
    pub ndim: usize,
    pub bias: f64,
    pub w: Vec<f64>,
    pub v: Vec<Vec<f64>>,
    pub l0: f64,
    pub l1: f64,
    pub l2: f64,
}

impl FM {
    pub fn new(k: &usize, ndim: &usize, l0: i32, l1: i32, l2: i32) -> FM {
        let mu: f64 = 0.0;
        let sigma: f64 = 0.01;
        let bias = gen_rand_val(mu, sigma);
        let w = gen_rand_vec(*ndim, mu, sigma);
        let v = gen_rand_mat(*ndim, *k, mu, sigma);
        FM {
            k: *k,
            ndim: *ndim,
            bias: bias,
            w: w,
            v: v,
            l0: l0 as f64,
            l1: l1 as f64,
            l2: l2 as f64,
        }
    }

    fn pointwise_sum(&self, vec: &Vec<f64>) -> f64 {
        inner_prod(&self.w, vec)
    }

    fn pairwise_sum(&self, vec: &Vec<f64>) -> f64 {
        /* memoization */
        let mut table = vec![];
        for f in 0..self.k {
            let mut partial_interaction = vec![];
            for i in 0..self.ndim {
                partial_interaction.push(self.v[i][f] * vec[i]);
            }
            table.push(partial_interaction);
        }

        /* calculation */
        let mut sum: f64 = 0.0;
        for f in 0..self.k {
            let mut first_sum: f64 = 0.0;
            let mut second_sum: f64 = 0.0;
            for i in 0..self.ndim {
                first_sum += table[f][i];
                second_sum += pow(table[f][i], 2);
            }
            sum += pow(first_sum, 2) - second_sum;
        }
        0.5 * sum
    }

    pub fn predict_one(&self, vec: &Vec<f64>) -> f64 {
        self.bias + self.pointwise_sum(vec) + self.pairwise_sum(vec)
    }

    pub fn predict(&self, features: &Vec<&Vec<f64>>) -> Vec<f64> {
        let mut t_pred = vec![];
        for i in 0..features.len() {
            let x = &features[i];
            t_pred.push(self.predict_one(x));
        }
        t_pred
    }

    pub fn fit(&mut self, features: &Vec<&Vec<f64>>, t: &Vec<f64>, num_iter: i32) {
        let eta: f64 = 0.001;

        for _epoch in 0..num_iter {
            for j in 0..t.len() {
                let x = &features[j];
                let t_pred = self.predict_one(x);
                let residue = t[j] - t_pred;

                /* updated bias caclulation */
                let updated_bias = self.bias + eta * (residue - self.l0 * self.bias);

                /* updated w calculation */
                let mut updated_w = vec![];
                for i in 0..self.ndim {
                    let updated_w_i = self.w[i] + eta * (residue * x[i] - self.l1 * self.w[i]);
                    updated_w.push(updated_w_i);
                }

                /* updated v calculation */
                let mut updated_v = vec![];
                for f in 0..self.k {
                    /* f behaves constantly in this loop */
                    let mut partial_sum: f64 = 0.0;
                    for h in 0..self.ndim {
                        partial_sum += self.v[h][f] * x[h];
                    }
                    /* ith factorized vector */
                    let mut updated_v_i = vec![];
                    for i in 0..self.ndim {
                        let nabra = x[i] * partial_sum - self.v[i][f] * pow(x[i], 2);
                        let updated_v_i_f: f64 = self.v[i][f] + eta * (residue * nabra - self.l2 * self.v[i][f]);
                        updated_v_i.push(updated_v_i_f);
                    }
                    updated_v.push(updated_v_i);
                }
                /* transpose updated_v, whose shape is k x ndim & shadow it */
                let updated_v = transpose(&updated_v);

                /* parameter updation */
                self.bias = updated_bias;
                self.w = updated_w;
                self.v = updated_v;
            }
        }
    }
}
