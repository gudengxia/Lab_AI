use std::usize;
use serde::Deserialize;
use candle_core::{Result, Tensor};
//use candle_nn::{loss, ops, Linear, Module, VarBuilder, VarMap, Optimizer};
//use anyhow::Error;
use super::*;
pub struct Dataset{
    pub x_train: Tensor,
    pub y_train: Tensor,
    pub x_test: Tensor,
    pub y_test: Tensor,
}

impl Dataset{
    pub fn new(train_size: usize, test_size: usize) -> Result<Self>{
        let n = train_size + test_size;
        let w_vec = [3.0, -2.0].to_vec();
        let b_vec = vec![2.0; n];
        let w = Tensor::from_vec(w_vec, (2, 1), &device)?;
        let b = Tensor::from_vec(b_vec, (n, 1), &device)?;
        let noise = Tensor::from_vec(vec![0.001f64; n], (n, 1), &device)?;
        let term = Tensor::randn(0., 1.0, (n, 1), &device)?.mul(&noise)?;
        let x = Tensor::randn(0., 1.0, (n, 2), &device)?;
        let y = (x.matmul(&w))?.add(&term)?.add(&b)?;
        let x_train = x.narrow(0, 0, train_size)?;
        let y_train = y.narrow(0, 0, train_size)?;
        let x_test = x.narrow(0, train_size, test_size)?;
        let y_test = y.narrow(0, train_size, test_size)?;
        return Ok(Self{x_train, y_train, x_test, y_test})
    }

    pub fn get_dataloader(&self) ->&Self{
        return &self;
    }

    pub fn read_from_csv(file: &str)->Result<Self>{
        #[derive(Debug, Deserialize)]
        struct Record{
            pub x0: f64,
            pub x1: f64,
            pub y: f64
        }
        let mut reader = csv::Reader::from_path(file).expect("read file error.");

        let mut x_vec = Vec::<f64>::new();
        let mut y_vec = Vec::<f64>::new();
        for item in reader.deserialize() {
            let record: Record = item.expect("convert error.");
            x_vec.push(record.x0);
            x_vec.push(record.x1);
            y_vec.push(record.y);
        }
        let n = x_vec.len();
        let x = Tensor::from_vec(x_vec, (n, 2), &device)?;
        let y = Tensor::from_vec(y_vec, (n, 1), &device)?;
        let x_train = x.narrow(0, 0, 1024)?;
        let y_train = y.narrow(0, 0, 1024)?;
        let x_test = x.narrow(0, 1024, 128)?;
        let y_test = y.narrow(0, 1024, 128)?;
        Ok(Self{x_train, y_train, x_test, y_test})
    }
}



