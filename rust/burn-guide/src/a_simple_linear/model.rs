
use burn::{
    nn::{
        Linear, LinearConfig,
        loss::{MseLoss, Reduction:: Mean},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use crate::a_simple_linear::data::RegressionBatch;

#[derive(Module, Debug)]
pub struct RegressionModel<B: Backend> {
    ln: Linear<B>,
}

#[derive(Config, Debug)]
pub struct RegressionModelConfig {
}

impl RegressionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
        let ln = LinearConfig::new(2, 1)
            .with_bias(true)
            .init(device);
        RegressionModel {
            ln}
    }
}

impl<B: Backend> RegressionModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.ln.forward(input);
        x
    }

    pub fn forward_step(&self, item: RegressionBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze_dim(1);
        //let targets: Tensor<B, 2> = item.y.unsqueeze_dim(0);
        let output: Tensor<B, 2> = self.forward(item.inputs);
        //println!("batch {:?}, {:?}", item.x.clone(), targets.clone());
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);
        //let loss = MseLoss::new().forward(output.clone(), targets.clone(), Sum);
        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<RegressionBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: RegressionBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);
    
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<RegressionBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: RegressionBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}