use image::{GrayImage, ImageBuffer, Luma};

fn add(a:&[Vec<f32>],b:&[Vec<f32>])->Vec<Vec<f32>>{let n=a.len();let mut r=vec![vec![0.0;n];n];for i in 0..n{for j in 0..n{r[i][j]=a[i][j]+b[i][j];}}r}
fn sub(a:&[Vec<f32>],b:&[Vec<f32>])->Vec<Vec<f32>>{let n=a.len();let mut r=vec![vec![0.0;n];n];for i in 0..n{for j in 0..n{r[i][j]=a[i][j]-b[i][j];}}r}
fn naive_square_multiply(a:&[Vec<f32>],b:&[Vec<f32>])->Vec<Vec<f32>>{let n=a.len();let mut c=vec![vec![0.0;n];n];for i in 0..n{for j in 0..n{let mut s=0.0;for k in 0..n{s+=a[i][k]*b[k][j];}c[i][j]=s;}}c}
fn strassen_square_multiply(a:&[Vec<f32>],b:&[Vec<f32>])->Vec<Vec<f32>>{let n=a.len();if n<=2{return naive_square_multiply(a,b);}let m=n/2;let mut a11=vec![vec![0.0;m];m];let mut a12=vec![vec![0.0;m];m];let mut a21=vec![vec![0.0;m];m];let mut a22=vec![vec![0.0;m];m];let mut b11=vec![vec![0.0;m];m];let mut b12=vec![vec![0.0;m];m];let mut b21=vec![vec![0.0;m];m];let mut b22=vec![vec![0.0;m];m];
for i in 0..m{for j in 0..m{a11[i][j]=a[i][j];a12[i][j]=a[i][j+m];a21[i][j]=a[i+m][j];a22[i][j]=a[i+m][j+m];b11[i][j]=b[i][j];b12[i][j]=b[i][j+m];b21[i][j]=b[i+m][j];b22[i][j]=b[i+m][j+m];}}
let m1=strassen_square_multiply(&add(&a11,&a22),&add(&b11,&b22));
let m2=strassen_square_multiply(&add(&a21,&a22),&b11);
let m3=strassen_square_multiply(&a11,&sub(&b12,&b22));
let m4=strassen_square_multiply(&a22,&sub(&b21,&b11));
let m5=strassen_square_multiply(&add(&a11,&a12),&b22);
let m6=strassen_square_multiply(&sub(&a21,&a11),&add(&b11,&b12));
let m7=strassen_square_multiply(&sub(&a12,&a22),&add(&b21,&b22));
let mut c=vec![vec![0.0;n];n];
for i in 0..m{for j in 0..m{
c[i][j]=m1[i][j]+m4[i][j]-m5[i][j]+m7[i][j];
c[i][j+m]=m3[i][j]+m5[i][j];
c[i+m][j]=m2[i][j]+m4[i][j];
c[i+m][j+m]=m1[i][j]-m2[i][j]+m3[i][j]+m6[i][j];
}}
c}
fn naive_multiply(a:&[Vec<f32>],b:&[Vec<Vec<f32>>])->Vec<Vec<f32>>{let m=a.len();let k=a[0].len();let n=b[0].len();let mut c=vec![vec![0.0;n];m];for i in 0..m{for j in 0..n{let mut s=0.0;for x in 0..k{s+=a[i][x]*b[x][j];}c[i][j]=s;}}c}
fn multiply_strassen(a:&[Vec<f32>],b:&[Vec<f32>])->Vec<f32>{if a.is_empty(){return vec![];}if a[0].is_empty(){return vec![];}let m=a.len();let k=a[0].len();let n=1;let mut bb=vec![vec![0.0;1];k];for i in 0..k{bb[i][0]=b[i];}
if m==k&&k==n{return (strassen_square_multiply(a,&bb)).iter().map(|row|row[0]).collect();}
let c=naive_multiply(a,&bb);c.iter().map(|row|row[0]).collect()}
pub fn convolve(image:&GrayImage,kernel:&[&[f32]])->GrayImage{let (kh,kw)=(kernel.len(),kernel[0].len());let (width,height)=image.dimensions();let output_width=width-kw as u32+1;let output_height=height-kh as u32+1;let mut output=ImageBuffer::new(output_width,output_height);let mut mat=vec![vec![0.0;(kh*kw)];(output_height*output_width)as usize];for y in 0..output_height{for x in 0..output_width{let row=(y*output_width+x)as usize;let mut idx=0;for ky in 0..kh{for kx in 0..kw{mat[row][idx]=image.get_pixel(x+kx as u32,y+ky as u32)[0]as f32;idx+=1;}}}}
let mut ker=vec![0.0;kh*kw];let mut idx=0;for ky in 0..kh{for kx in 0..kw{ker[idx]=kernel[ky][kx];idx+=1;}}
let res=multiply_strassen(&mat,&ker);for y in 0..output_height{for x in 0..output_width{let val=res[(y*output_width+x)as usize].clamp(0.0,255.0)as u8;output.put_pixel(x,y,Luma([val]));}}
output}