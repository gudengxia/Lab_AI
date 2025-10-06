use candle_core::{DType, Result, Tensor, D};
use candle_nn::{loss::nll, Linear, optim, Module, Optimizer, VarBuilder, VarMap};
use super::*;
#[derive(Clone)]

pub  struct ExPerceptron {
        ln1: Linear,
        ln2: Linear,
        ln3: Linear,
    }

impl ExPerceptron {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IN_DIM, LAYER1_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_SIZE, LAYER2_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_SIZE, OUT_DIM + 1, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}

#[derive(Clone)]
pub struct MultiLevelPerceptron {
    pub params: VarMap,
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}


impl MultiLevelPerceptron {
    pub fn new() -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let ln1 = candle_nn::linear(IN_DIM, LAYER1_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_SIZE, LAYER2_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_SIZE, OUT_DIM + 1, vs.pp("ln3"))?;
        Ok(Self {params: varmap, ln1, ln2, ln3 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;   
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?; 
        self.ln3.forward(&xs)
    }

    pub fn loss(&self, y_hat: &Tensor, y: &Tensor)->Result<Tensor>{
        return nll(y_hat, y); //mse
    }

    pub fn configure_optimizers(&self)->Result<optim::SGD>{
        return candle_nn::SGD::new(self.params.all_vars(), LEARNING_RATE);
    }

    pub fn validate(&self)->Result<()>{
        let real_world_votes: Vec<u32> = vec![
            13, 22,
        ];

        let tensor_test_votes = Tensor::from_vec(real_world_votes.clone(), (1, IN_DIM), &device)?.to_dtype(DType::F32)?;

        let final_result = self.forward(&tensor_test_votes)?;

        let result = final_result
            .argmax(D::Minus1)?
            .to_dtype(DType::F32)?
            .get(0).map(|x| x.to_scalar::<f32>())??;
        println!("real_life_votes: {:?}", real_world_votes);
        println!("neural_network_prediction_result: {:?}", result);
        Ok(())
    }
}

