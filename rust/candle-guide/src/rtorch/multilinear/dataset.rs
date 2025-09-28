use candle_core::{Result, Tensor, DType};
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
    pub fn new() -> Result<Self>{
        let x_vec: Vec<u32> = vec![
            15, 10,
            10, 15,
            5, 12,
            30, 20,
            16, 12,
            13, 25,
            6, 14,
            31, 21,
        ];
        let x_train = Tensor::from_vec(x_vec.clone(), (x_vec.len() / IN_DIM, IN_DIM), &device)?.to_dtype(DType::F32)?;

        let y_vec: Vec<u32> = vec![
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
        ];
        let y_train = Tensor::from_vec(y_vec.clone(), y_vec.len() / OUT_DIM, &device)?;

        let x_test_vec: Vec<u32> = vec![
            13, 9,
            8, 14,
            3, 10,
        ];
        let x_test = Tensor::from_vec(x_test_vec.clone(), (x_test_vec.len() / IN_DIM, IN_DIM), &device)?.to_dtype(DType::F32)?;

        let y_test_vec: Vec<u32> = vec![
            1,
            0,
            0,
        ];
        let y_test = Tensor::from_vec(y_test_vec.clone(), y_test_vec.len(), &device)?;

        Ok(Self{
            x_train: x_train,
            y_train: y_train,
            x_test: x_test,
            y_test: y_test
        })
    }

    pub fn get_dataloader(&self) ->&Self{
        return &self;
    }
}


mod test
{
    use super::*;

    #[tokio::test]
    async fn test_tensor()
    {
        const device: Device = Device::Cpu;
        let x = Tensor::from_vec([1f32, 2.0, 4.0, 3.0, 2.0, 1.0].to_vec(), (3, 2), &device).expect("error.");
        let z = x.t().expect("error!");
        let y = Tensor::from_vec([1f32, 2.0].to_vec(), (2, 1), &device).expect("error.");
        //let y = Tensor::from_vec([1f32, 2.0].to_vec(), 2, &device).expect("error.");
        //error: shape mismatch in matmul, rhs[3, 2], rhs:[1]
        let m = Tensor::matmul(&x, &z).expect("mul error.");
        let n = Tensor::matmul(&x, &y).expect("mult error.");
        println!("{:?}", z.shape());
        println!("{:?}", m.shape());
        println!("{:?}", n.shape());
    }
}