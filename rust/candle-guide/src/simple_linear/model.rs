use candle_core::{Module, DType, Result, Tensor};
use candle_nn::{loss::mse, optim, Linear, Optimizer, VarBuilder, VarMap};
use super::*;

#[derive(Clone)]
pub  struct ALinearPerceptron {
    params: VarMap,
    ln: Linear
}


impl ALinearPerceptron {
    pub fn new() -> Result<Self> {
        let params = VarMap::new();
        let vs = VarBuilder::from_varmap(&params, DType::F64, &device);
        let ws = vs.get_with_hints((1, 2), "weight",candle_nn::init::ZERO)?;
        let bs = vs.get_with_hints(1, "bias", candle_nn::init::ZERO)?;
        let ln = Linear::new(ws, Some(bs));
        Ok(Self { params, ln})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln.forward(x)?;
        Ok(x) 
    }

    pub fn loss(&self, y_hat: &Tensor, y: &Tensor)->Result<Tensor>{
        return mse(y_hat, y); //mse
    }

    pub fn configure_optimizers(&self)-> Result<optim::SGD>{
        let r = candle_nn::SGD::new(self.params.all_vars(), 0.0005)?;
        return Ok(r);
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

