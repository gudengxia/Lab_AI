use candle_core::{Result, DType};
use candle_nn::Optimizer;
use rand::rng;
use rand::prelude::SliceRandom;
use crate::simple_linear::{dataset::Dataset, model::ALinearPerceptron};
use super::*;

pub struct Trainer{
    epochs: usize,
    batch_size: usize
}

impl Trainer {
    pub fn new(epochs: usize, batch_size: usize) -> Trainer{
        Trainer {epochs, batch_size}
    }

    pub fn fit(&self, model: &ALinearPerceptron, data: &Dataset)->Result<()>{
        let mut sgd = model.configure_optimizers()?;
        let x_train = data.x_train.to_device(&device)?.to_dtype(DType::F64)?.to_device(&device)?;
        let y_train = data.y_train.to_device(&device)?.to_dtype(DType::F64)?.to_device(&device)?;
        let x_test = data.x_test.to_device(&device)?.to_dtype(DType::F64)?;
        let y_test = data.y_test.to_device(&device)?.to_dtype(DType::F64)?;
        let n = x_train.dim(0)?;
        let nbatches = n / self.batch_size;
        let mut idxs = (0..nbatches).collect::<Vec<usize>>();
        for epoch in 0..self.epochs{
            let mut sum_loss = 0.0f64;
            idxs.shuffle(&mut rng());
            println!("1");
            for idx in idxs.iter(){
                let _x = x_train.narrow(0, idx*self.batch_size, self.batch_size)?;
                let _y = y_train.narrow(0, idx*self.batch_size, self.batch_size)?;
                let y_hat = model.forward(&_x)?;
                let loss = model.loss(&y_hat, &_y)?;               
                sgd.backward_step(&loss)?;
                sum_loss += loss.to_vec0::<f64>()?;               
            }
            
            let avg_loss = sum_loss / (nbatches as f64);
            let y_hat = model.forward(&x_test)?;
            //let accuracy = model.loss(&y_hat, &y_test)?;
            let accuracy = model.loss(&y_hat, &y_test)?.to_vec0::<f64>()? / (y_test.dim(0)? as f64); 
            println!("Epoch {:?} -- avg_loss:{:?}, avg_accuracy:{:?}", epoch, avg_loss, accuracy);
        }
        Ok(())
    }
}