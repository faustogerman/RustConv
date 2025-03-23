use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;

fn add(a:&[f32],b:&[f32],n:usize)->Vec<f32>{let mut r=vec![0.;n];r.par_iter_mut().enumerate().for_each(|(i,x)|*x=a[i]+b[i]);r}
fn subm(a:&[f32],b:&[f32],n:usize)->Vec<f32>{let mut r=vec![0.;n];r.par_iter_mut().enumerate().for_each(|(i,x)|*x=a[i]-b[i]);r}
fn naive_mul(a:&[f32],ar:usize,ac:usize,b:&[f32],bc:usize)->Vec<f32>{let mut c=vec![0.;ar*bc];(0..ar).into_par_iter().for_each(|i|{for j in 0..bc{let mut s=0.;for k in 0..ac{s+=a[i*ac+k]*b[k*bc+j];}c[i*bc+j]=s;}});c}
fn get_submatrix(a:&[f32],n:usize,r1:usize,c1:usize,h:usize,w:usize)->Vec<f32>{let mut r=vec![0.;h*w];for i in 0..h{for j in 0..w{r[i*w+j]=a[(r1+i)*n+(c1+j)];}}r}
fn put_submatrix(c:&mut[f32],n:usize,r1:usize,c1:usize,h:usize,w:usize,sm:&[f32]){for i in 0..h{for j in 0..w{c[(r1+i)*n+(c1+j)]=sm[i*w+j];}}}
fn strassen(a:&[f32],b:&[f32],n:usize,th:usize)->Vec<f32>{
    if n<=th{return naive_mul(a,n,n,b,n);}
    let hn=n/2;
    let a11=get_submatrix(a,n,0,0,hn,hn);
    let a12=get_submatrix(a,n,0,hn,hn,hn);
    let a21=get_submatrix(a,n,hn,0,hn,hn);
    let a22=get_submatrix(a,n,hn,hn,hn,hn);
    let b11=get_submatrix(b,n,0,0,hn,hn);
    let b12=get_submatrix(b,n,0,hn,hn,hn);
    let b21=get_submatrix(b,n,hn,0,hn,hn);
    let b22=get_submatrix(b,n,hn,hn,hn,hn);
    let (m1,m2)=rayon::join(||{let x=add(&a11,&a22,hn*hn);let y=add(&b11,&b22,hn*hn);strassen(&x,&y,hn,th)},||{let x=add(&a21,&a22,hn*hn);strassen(&x,&b11,hn,th)});
    let (m3,m4)=rayon::join(||{let y=subm(&b12,&b22,hn*hn);strassen(&a11,&y,hn,th)},||{let y=subm(&b21,&b11,hn*hn);strassen(&a22,&y,hn,th)});
    let (m5,m6,m7)={
        let r1=||{let x=add(&a11,&a12,hn*hn);strassen(&x,&b22,hn,th)};
        let r2=||{let x=subm(&a21,&a11,hn*hn);let y=add(&b11,&b12,hn*hn);strassen(&x,&y,hn,th)};
        let r3=||{let x=subm(&a12,&a22,hn*hn);let y=add(&b21,&b22,hn*hn);strassen(&x,&y,hn,th)};
        let (p5,p6)=rayon::join(r1,r2);
        let p7=r3();
        (p5,p6,p7)
    };
    let mut c=vec![0.;n*n];
    let c11_1=add(&m1,&m4,hn*hn);
    let c11_2=subm(&c11_1,&m5,hn*hn);
    let c11=add(&c11_2,&m7,hn*hn);
    let c12=add(&m3,&m5,hn*hn);
    let c21=add(&m2,&m4,hn*hn);
    let c22_1=add(&m1,&m3,hn*hn);
    let c22_2=subm(&c22_1,&m2,hn*hn);
    let c22=add(&c22_2,&m6,hn*hn);
    put_submatrix(&mut c,n,0,0,hn,hn,&c11);
    put_submatrix(&mut c,n,0,hn,hn,hn,&c12);
    put_submatrix(&mut c,n,hn,0,hn,hn,&c21);
    put_submatrix(&mut c,n,hn,hn,hn,hn,&c22);
    c
}
fn mul(a:&[f32],ar:usize,ac:usize,b:&[f32],br:usize,bc:usize)->Vec<f32>{
    if ac!=br{return vec![];}
    let sq=ar==ac&&br==bc&&ar==br;
    if sq{
        let mut p=1;while p<ar{p<<=1;}if p==ar{return strassen(a,b,ar,64);}
    }
    naive_mul(a,ar,ac,b,bc)
}
fn im2col(image:&GrayImage,kh:usize,kw:usize,oh:usize,ow:usize)->Vec<f32>{
    let mut r=vec![0.;(oh*ow*kh*kw) as usize];
    r.par_chunks_exact_mut((kh*kw) as usize).enumerate().for_each(|(i,row)|{
        let y=i/ow;let x=i%ow;
        for ky in 0..kh{for kx in 0..kw{
            row[ky*kw+kx]=image.get_pixel(x as u32 +kx as u32,y as u32+ky as u32)[0]as f32;
        }}
    });
    r
}
pub fn convolve(image:&GrayImage,kernel:&[&[f32]])->GrayImage{
    let (kh,kw)=(kernel.len(),kernel[0].len());
    let (width,height)=image.dimensions();
    let ow=width-kw as u32+1;let oh=height-kh as u32+1;
    let a=im2col(image,kh,kw,oh as usize,ow as usize);
    let mut b=vec![0.;kh*kw];for i in 0..kh{for j in 0..kw{b[i*kw+j]=kernel[i][j];}}
    let c=mul(&a,(ow*oh)as usize,kh*kw,&b,kh*kw,1);
    let mut buffer=vec![0u8;(ow*oh)as usize];
    buffer.par_iter_mut().enumerate().for_each(|(i,x)|{
        *x=c[i].clamp(0.,255.)as u8;
    });
    let mut output=ImageBuffer::new(ow,oh);
    for(i,p)in buffer.iter().enumerate(){
        let x=(i%ow as usize)as u32;let y=(i/ow as usize)as u32;
        output.put_pixel(x,y,Luma([*p]));
    }
    output
}