pub mod data;
pub mod discriminator;
pub mod generator;
pub mod training;
pub mod inference;
use burn::tensor::Distribution::Normal;
use burn::tensor::{self, Tensor};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::prelude::Backend;

const BATCH_SIZE: usize = 64;
const NOISE_DIM: usize = 64;

fn rand_nosie<B: Backend>(dev: &B::Device)-> Tensor<B, 4>{
    let dis = Normal(0.0, 1.0);
    let t = tensor::Tensor::<B, 2>::random([BATCH_SIZE, NOISE_DIM], dis, dev);
    t.reshape([BATCH_SIZE, 1, 1, NOISE_DIM])
} 

fn bce_with_binary_entroy<const D: usize, B: Backend>(y_hat: &Tensor<B, D>, y: &Tensor<B, D>, dev: &B::Device)
-> Tensor<B, 1>
{
    let logits = burn::tensor::activation::sigmoid(y_hat.clone());
    let bce = BinaryCrossEntropyLossConfig::new().init(dev);
    let r = bce.forward(logits, y.clone().int());
    r
}