use burn::config::Config;
use burn::nn::loss::{CrossEntropyLoss, MseLoss, Reduction::Mean};
use crate::regression::model::RegressionModelConfig;
use crate::regression::model::RegressionModel;
use burn::optim::AdamConfig;
//use burn::train::LearningStrategy;
use burn::{
    tensor::backend::AutodiffBackend,
    optim::{GradientsParams,Optimizer},
};
use super::dataset::HousingData;
use rand::rng;
use rand::prelude::SliceRandom;
use rgb::RGB8;
use textplots::{Chart, ColorPlot, Shape};
#[derive(Config)]
pub struct LearnerConfig {
    #[config(default = 3)]
    pub num_epochs: usize,

    #[config(default = 1)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    pub optimizer: AdamConfig,

    #[config(default = 256)]
    pub batch_size: usize,

    #[config(default = 1e-3)]
    pub lr: f64,
}


pub fn fit<B: AutodiffBackend>( device: B::Device) {
    // Create the configuration.
    let optimizer = AdamConfig::new();
    let config = LearnerConfig::new(optimizer);
    let mut model: RegressionModel<B>= RegressionModelConfig::new().init(&device);
    B::seed(config.seed);

    // Create the model and optimizer.
    let dataset = HousingData::new(&device);   

    let x_train = dataset.x_train;
    let y_train = dataset.y_train;
    let n = x_train.shape().dims[0];

    let n_batches = n / config.batch_size;
    let mut idxs = (0..n_batches).collect::<Vec<usize>>();
    let mut optim = config.optimizer.init();
    for _epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        idxs.shuffle(&mut rng());
        for idx in idxs.iter() {

            let x = x_train.clone().narrow(0, idx*config.batch_size, config.batch_size);
            let y = y_train.clone().narrow(0, idx*config.batch_size, config.batch_size);
            let y_hat = model.forward(x);  
            let loss = MseLoss::new().forward(y_hat.clone(), y.clone().unsqueeze_dim(1), Mean);

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);              
        }
    }

    let x_test = dataset.x_test;
    let y_test = dataset.y_test;
    let y_hat = model.forward(x_test);

    // Display the predicted vs expected values
    let y_hat = y_hat.squeeze_dims::<1>(&[1]).into_data();
    let y = y_test.into_data();

    let points = y_hat
        .iter::<f32>()
        .zip(y.iter::<f32>())
        .collect::<Vec<_>>();

    println!("Predicted vs. Expected Median House Value (in 100,000$)");
    Chart::new_with_y_range(120, 60, 0., 5., 0., 5.)
        .linecolorplot(
            &Shape::Points(&points),
            RGB8 {
                r: 255,
                g: 85,
                b: 85,
            },
        )
        .display();

    // Print a single numeric value as an example
    println!("Predicted {} Expected {}", points[0].0, points[0].1);

}