use crate::a_simple_linear::data::RegressionDataset;
use crate::a_simple_linear::model::RegressionModelConfig;
use burn::optim::SgdConfig;
//use burn::train::LearningStrategy;
use crate::a_simple_linear::data::RegressionBatcher;
use crate::a_simple_linear::model::RegressionModel;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record:: {NoStdTrainingRecorder,CompactRecorder},
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

#[derive(Config)]
pub struct TrainConfig {
    #[config(default = 3)]
    pub num_epochs: usize,

    #[config(default = 1)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    pub optimizer: SgdConfig,

    #[config(default = 64)]
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config
    let optimizer = SgdConfig::new();
    let config = TrainConfig::new(optimizer);
    let model: RegressionModel<B> = RegressionModelConfig::new().init(&device);
    B::seed(config.seed);

    // Define train/valid datasets and dataloaders
    let train_dataset = RegressionDataset::train();
    let valid_dataset = RegressionDataset::validation();

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = RegressionBatcher::new(device.clone());

    let batcher_test = RegressionBatcher::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
	//learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), 1e-5);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}