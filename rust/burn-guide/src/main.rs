use burn_guide::a_simple_linear::{training::train, inference::infer, learner::run, trainer::fit};
use burn::backend::Autodiff;
use burn_candle::{Candle, CandleDevice};
use burn_ndarray::NdArrayDevice;
use std::time:: Instant;



fn main() {
    /*** candle backend***/
    //type MyBackend = Candle<f32>;
    //let device = CandleDevice::Cpu;
    ////debug: adaptive_avg_pool2 is not supported by Candle
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
    let artifact_dir = "./tmp/guide";
    let tm = Instant::now();
    train::<MyAutodiffBackend>(
        artifact_dir,
        device.clone(),
    );
    let diff = tm.elapsed().as_micros();
    println!("Consume time: {} ms.", diff);
    infer::<MyBackend>(
        artifact_dir,
        device.clone(),
    );
    run::<MyAutodiffBackend>(device.clone());
    fit::<MyAutodiffBackend>(device.clone());
}