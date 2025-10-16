use rand::prelude::SliceRandom;
use rand::rng;
use crate::simple_linear::dataset::Dataset;
use super::*;
use candle_core::{DType, Result, Tensor};
use candle_nn::{loss::mse, optim, Linear, Module, Optimizer, VarBuilder, VarMap};
use std::time::{Duration, Instant};

// a very import lesson here is that a Linear perception should have two parts: w and b
fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder)->Result<Linear>{
    let ws = vs.get_with_hints((out_dim, in_dim), "weight",candle_nn::init::ZERO)?;
    let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}
struct SLP{
    ln: Linear,
}

impl SLP {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        //let ln = candle_nn::linear(2, 1, vs.pp("ln"))?;
        let ln = linear_z(2, 1, vs)?;
        Ok(Self {ln})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln.forward(x)?;
        Ok(x) 
    }

    pub fn validate(&self)->Result<()>{
        let x_valid_vec: Vec<f64> = vec![
            0.9, 0.5
        ];

        let x_valid = Tensor::from_vec(x_valid_vec.clone(), (1, 2), &device)?.to_dtype(DType::F64)?.to_device(&device)?;
        let y_valid = self.forward(&x_valid)?;
        
        //println!("x_valid: {:?}", x_valid.to_vec0::<f32>()?);
        println!("y_valid: {:?}", y_valid.reshape(())?.to_scalar::<f64>()?);
        Ok(())
    }
}
pub struct Learner{
    epochs: usize,
    batch_size: usize,
    lr: f64
}

impl Learner{
    pub fn new(epochs: usize, batch_size: usize) -> Learner{
        let lr = 0.0005;
        Learner {epochs, batch_size, lr}
    }

    pub fn train(&self, m: Dataset) -> Result<()> {
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
        let model = SLP::new(vs.clone())?;
        let mut sgd = optim::SGD::new(varmap.all_vars(), self.lr)?;

        let x_train = m.x_train.to_device(&device)?.to_dtype(DType::F64)?.to_device(&device)?;
        let y_train = m.y_train.to_device(&device)?.to_dtype(DType::F64)?.to_device(&device)?;
        let x_test = m.x_test.to_device(&device)?.to_dtype(DType::F64)?;
        let y_test = m.y_test.to_device(&device)?.to_dtype(DType::F64)?;
        let n = x_train.dim(0)?;
        let nbatches = n / self.batch_size;
        let mut idxs = (0..nbatches).collect::<Vec<usize>>();
        
        let mut total_time = 0u128;
        for epoch in 0..self.epochs{
            let mut sum_loss = 0.0f64;
            idxs.shuffle(&mut rng());
            let start = Instant::now();
            for idx in idxs.iter(){
                let _x = x_train.narrow(0, idx*self.batch_size, self.batch_size)?;
                let _y = y_train.narrow(0, idx*self.batch_size, self.batch_size)?;
                let y_hat = model.forward(&_x)?;
                let loss = mse(&y_hat, &_y)?;               
                sgd.backward_step(&loss)?;
                sum_loss += loss.to_vec0::<f64>()?;               
            }
            let diff = start.elapsed().as_micros();
            println!("Epoch consumes: {}", diff);
            total_time += diff;
            let avg_loss = sum_loss / (nbatches as f64);
            let y_hat = model.forward(&x_test)?;
            //let accuracy = model.loss(&y_hat, &y_test)?;
            let accuracy = mse(&y_hat, &y_test)?.to_vec0::<f64>()? / (y_test.dim(0)? as f64); 
            println!("Epoch {:?} -- avg_loss:{:?}, avg_accuracy:{:?}\n", epoch, avg_loss, accuracy);     
        }
        println!("Consume {:?} microseconds totoally.", total_time);
        let _ = model.validate();
        Ok(())
    }
}