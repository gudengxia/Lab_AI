use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::AutodiffModule,
    nn::loss::{MseLoss, Reduction::Mean, Reduction::Sum},
    optim::{AdamConfig, GradientsParams, Optimizer, Sgd, SgdConfig},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::rng;
use rand::prelude::SliceRandom;
use crate::a_simple_linear::model::RegressionModelConfig;
use crate::a_simple_linear::data::{RegressionBatch, RegressionBatcher, RegressionDataset, train_data};
#[derive(Config)]
pub struct LearnerConfig {
    #[config(default = 3)]
    pub num_epochs: usize,
    
    #[config(default = 64)]
    pub batch_size: usize,
    
    #[config(default = 1)]
    pub num_workers: usize,
    
    #[config(default = 42)]
    pub seed: u64,
    
    #[config(default = 1e-4)]
    pub lr: f64,
    
    pub model: RegressionModelConfig,
    
    pub optimizer: SgdConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Create the configuration.
    let config_model = RegressionModelConfig::new();
    let config_optimizer = SgdConfig::new();
    let config = LearnerConfig::new(config_model, config_optimizer);

    // Create the model and optimizer.
    let mut model = config.model.init::<B>(&device);
    

    let train_data = train_data::new(&device);
    let x_train = train_data.x;
    let y_train = train_data.y;
    let n = train_data.len;
    let n_batches = n / config.batch_size;
    let mut idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        idxs.shuffle(&mut rng());
        for idx in idxs.iter() {
            //let start = Instant::now();
            let x = x_train.clone().narrow(0, idx*config.batch_size, config.batch_size);
            let y = y_train.clone().narrow(0, idx*config.batch_size, config.batch_size);
            let y_hat = model.forward(x);  
            let loss = MseLoss::new().forward(y_hat.clone(), y.clone().unsqueeze_dim(1), Mean);

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            let mut optim = config.optimizer.init();
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);              
        }
    }

    let x = Tensor::<B,2>::from_data([[0.9026f32, 1.0f32],[1.0f32, 1.0f32]], &device);
    
    let y_hat = model.forward(x);

    // Print a single numeric value as an example
    println!("Learner Predicted y is {:?}", y_hat.into_data().into_vec::<f32>());
}
