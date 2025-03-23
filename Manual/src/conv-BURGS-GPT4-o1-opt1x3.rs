use image::{GrayImage, ImageBuffer, Luma};
use std::cmp::max;

fn pad_matrix(a: &Vec<Vec<f32>>, size: usize) -> Vec<Vec<f32>> {
    let mut padded = vec![vec![0.0; size]; size];
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            padded[i][j] = a[i][j];
        }
    }
    padded
}

fn add_matrices(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let m = a.len();
    let n = a[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

fn sub_matrices(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let m = a.len();
    let n = a[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

fn naive_multiply(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let m = a.len();
    let n = a[0].len();
    let p = b[0].len();
    let mut c = vec![vec![0.0; p]; m];
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

fn strassen(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = a.len();
    if n == 1 {
        return vec![vec![a[0][0] * b[0][0]]];
    }
    let k = n / 2;
    let mut a11 = vec![vec![0.0; k]; k];
    let mut a12 = vec![vec![0.0; k]; k];
    let mut a21 = vec![vec![0.0; k]; k];
    let mut a22 = vec![vec![0.0; k]; k];
    let mut b11 = vec![vec![0.0; k]; k];
    let mut b12 = vec![vec![0.0; k]; k];
    let mut b21 = vec![vec![0.0; k]; k];
    let mut b22 = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            a11[i][j] = a[i][j];
            a12[i][j] = a[i][j + k];
            a21[i][j] = a[i + k][j];
            a22[i][j] = a[i + k][j + k];
            b11[i][j] = b[i][j];
            b12[i][j] = b[i][j + k];
            b21[i][j] = b[i + k][j];
            b22[i][j] = b[i + k][j + k];
        }
    }
    let m1 = strassen(&add_matrices(&a11, &a22), &add_matrices(&b11, &b22));
    let m2 = strassen(&add_matrices(&a21, &a22), &b11);
    let m3 = strassen(&a11, &sub_matrices(&b12, &b22));
    let m4 = strassen(&a22, &sub_matrices(&b21, &b11));
    let m5 = strassen(&add_matrices(&a11, &a12), &b22);
    let m6 = strassen(&sub_matrices(&a21, &a11), &add_matrices(&b11, &b12));
    let m7 = strassen(&sub_matrices(&a12, &a22), &add_matrices(&b21, &b22));
    let c11 = add_matrices(&sub_matrices(&add_matrices(&m1, &m4), &m5), &m7);
    let c12 = add_matrices(&m3, &m5);
    let c21 = add_matrices(&m2, &m4);
    let c22 = add_matrices(&sub_matrices(&add_matrices(&m1, &m3), &m2), &m6);
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..k {
        for j in 0..k {
            c[i][j] = c11[i][j];
            c[i][j + k] = c12[i][j];
            c[i + k][j] = c21[i][j];
            c[i + k][j + k] = c22[i][j];
        }
    }
    c
}

fn is_power_of_two(x: usize) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

fn strassen_multiply(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let rows_b = b.len();
    let cols_b = b[0].len();
    if cols_a != rows_b {
        return vec![];
    }
    let n = max(rows_a, max(cols_a, max(rows_b, cols_b)));
    let mut s = 1;
    while s < n {
        s <<= 1;
    }
    if rows_a == cols_a && rows_a == rows_b && rows_b == cols_b && rows_a == s && is_power_of_two(s) {
        let pa = pad_matrix(a, s);
        let pb = pad_matrix(b, s);
        let pc = strassen(&pa, &pb);
        let mut c = vec![vec![0.0; cols_b]; rows_a];
        for i in 0..rows_a {
            for j in 0..cols_b {
                c[i][j] = pc[i][j];
            }
        }
        c
    } else {
        naive_multiply(a, b)
    }
}

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut mat = Vec::new();
    for y in 0..output_height {
        for x in 0..output_width {
            let mut row = Vec::new();
            for ky in 0..kh {
                for kx in 0..kw {
                    row.push(image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32);
                }
            }
            mat.push(row);
        }
    }
    let mut kern = Vec::new();
    for ky in 0..kh {
        for kx in 0..kw {
            kern.push(kernel[ky][kx]);
        }
    }
    let mut kern_2d = vec![vec![0.0; 1]; kh*kw];
    for i in 0..kh*kw {
        kern_2d[i][0] = kern[i];
    }
    let res = strassen_multiply(&mat.iter().map(|r| r.to_vec()).collect(), &kern_2d);
    let mut output = ImageBuffer::new(output_width, output_height);
    for (i, v) in res.iter().enumerate() {
        let val = v[0].clamp(0.0, 255.0) as u8;
        let y = (i as u32) / output_width;
        let x = (i as u32) % output_width;
        output.put_pixel(x, y, Luma([val]));
    }
    output
}