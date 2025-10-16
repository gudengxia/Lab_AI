use burn::{backend::Autodiff, tensor::backend::Backend};
use burn_regression::regression::{inference, training, learner};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
static ARTIFACT_DIR: &str = "./tmp/burn-example-regression";

pub fn run<B: Backend>(device: B::Device) {
    training::run::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
    inference::infer::<B>(ARTIFACT_DIR, device.clone());
    println!("\n\n");
    learner::fit::<Autodiff<B>>(device.clone());
}
fn main() {
    let device = NdArrayDevice::Cpu;
    run::<NdArray>(device.clone());
}