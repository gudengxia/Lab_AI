use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::{MseLoss, Reduction::Mean},
    optim::{SgdConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};

use crate::a_simple_linear::model::RegressionModelConfig;
use crate::a_simple_linear::data::{TrainData, RegressionBatcher, RegressionDataset};
#[derive(Config)]
pub struct TrainerConfig {
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

pub fn fit<B: AutodiffBackend>(device: B::Device) {
    // Create the configuration.
    let config_model = RegressionModelConfig::new();
    let config_optimizer = SgdConfig::new();
    let config = TrainerConfig::new(config_model, config_optimizer);

    //B::seed(&device, config.seed);

    // Create the model and optimizer.
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    let train_dataset = RegressionDataset::train();
    // Create the batcher.
    let batcher = RegressionBatcher::<B>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);


    // Iterate over our training and validation loop for X epochs.
    for _epochepoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            let y_hat = model.forward(batch.inputs);
            let loss = MseLoss::new()
                .forward(y_hat.clone(), batch.targets.clone().unsqueeze_dim(1), Mean);

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }        
    }

    //let x = Tensor::<B,2>::from_data([[0.9026f32, 1.0f32],[1.0f32, 1.0f32]], &device);
    //let x=x.to_device(&device);
    let x = TrainData::new(&device).x_test;
    let y_hat = model.forward(x);

    // Print a single numeric value as an example
    println!("Trainer Predicted y is {:?}", y_hat.into_data().iter::<f32>().collect::<Vec<_>>());
}
