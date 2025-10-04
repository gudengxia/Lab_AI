use burn_guide::{model::ModelConfig, training,training::{TrainingConfig}, inference::infer};
use burn::{
    backend::Autodiff, data::dataset::Dataset, optim::AdamConfig
};
//use burn_candle::{Candle, CandleDevice};
use burn_ndarray::NdArrayDevice;

fn main() {
    //type MyBackend = Wgpu<f32, i32>;
    //type MyBackend = Candle<f32, i32>; // Using Candle backend with f32 data and i32 indices
    /*** candle backend***/
    //type MyBackend = Candle;
    //let device = CandleDevice::Cpu;
    //debug: adaptive_avg_pool2 is not supported by Candle
    /*** candle backend***/
    /*** lbtorch backend***/
    //type MyBackend = Libtorch;
    //let device = Libtorch::Device::Cpu; 
    //error
    /*** nd_ndarray backend***/
    type MyBackend = burn::backend::NdArray<f32>;
    let device = NdArrayDevice::Cpu; 
    //let device = burn::backend::wgpu::WgpuDevice::default();
    
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let artifact_dir = "/tmp/guide";
   
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 64), AdamConfig::new()),
        device.clone(),
    );
    infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test().get(42).unwrap(),
    );
}