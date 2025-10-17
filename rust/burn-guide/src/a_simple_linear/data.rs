use burn::data::dataset::{Dataset, InMemDataset};
use burn::prelude::Backend;
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
pub const NUM_FEATURES: usize = 2;

// Pre-computed statistics for the housing dataset features
pub const FEATURES_MIN: [f32; NUM_FEATURES] = [-3.12656, -2.95123];
pub const FEATURES_MAX: [f32; NUM_FEATURES] = [3.752225, 2.938774];
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SingleItem {
    #[serde(rename = "x0")]
    pub x0: f32,
    #[serde(rename = "x1")]
    pub x1: f32,

    #[serde(rename = "y")]
    pub y: f32,
}

pub struct RegressionDataset {
    pub dataset: InMemDataset<SingleItem>,
}

impl RegressionDataset {
    pub fn new(is_train: bool) -> Result<Self, std::io::Error> {
        // Download dataset csv file
        let mut path = PathBuf::new();
        if is_train{
            path.push("./../../dataset/regression_train.csv");
        }
        else{
            path.push("./../../dataset/regression_test.csv");
        }

        // Build dataset from csv with tab ('\t') delimiter
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');
        let dataset: InMemDataset<SingleItem> = InMemDataset::from_csv(path, rdr).unwrap();
        let dataset = Self { dataset};

        Ok(dataset)
    }

    pub fn train()->Self{
        Self::new(true).expect("Load train_dataset error.")
    }

    pub fn test()->Self{
        Self::new(false).expect("Load test_dataset error.")
    }

    pub fn validation()->Self{
        Self::new(false).expect("Load test_dataset error.")
    }

}

// Implement the `Dataset` trait which requires `get` and `len`
impl Dataset<SingleItem> for RegressionDataset {
    fn get(&self, index: usize) -> Option<SingleItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub min: Tensor<B, 2>,
    pub max: Tensor<B, 2>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
        let min = Tensor::<B, 1>::from_floats(min, device).unsqueeze();
        let max = Tensor::<B, 1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    /// Normalizes the input according to the dataset min/max.
    pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }

    /// Returns a new normalizer on the given device.
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            min: self.min.clone().to_device(device),
            max: self.max.clone().to_device(device),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegressionBatcher<B: Backend> {
    normalizer: Normalizer<B>,
    device: B::Device
}

impl<B: Backend> RegressionBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        let normalizer = Normalizer::new(&device, &FEATURES_MIN, &FEATURES_MAX);
        Self{normalizer ,device}
    }
}

#[derive(Clone, Debug)]
pub struct RegressionBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> Batcher<B,SingleItem, RegressionBatch<B>> for RegressionBatcher<B> {
    fn batch(&self, items: Vec<SingleItem>, device: &B::Device) -> RegressionBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.x0,
                    item.x1,
                ],
                device,
            );
            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.normalizer.to_device(device).normalize(inputs);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.y], device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        RegressionBatch { inputs, targets}
    }
}

pub struct TrainData<B: Backend>{
    pub x: Tensor<B, 2>,
    pub y: Tensor<B, 1>,
    pub x_test: Tensor<B, 2>,
    pub len: usize
}

impl<B: Backend> TrainData<B> {
    pub fn new(device: &B::Device) -> Self{
        let dataset = RegressionDataset::new(true).unwrap().dataset;
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in dataset.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.x0,
                    item.x1,
                ],
                device,
            );
            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs.clone(), 0);
        let normalizer = Normalizer::new(device, &FEATURES_MIN, &FEATURES_MAX);
        let x = normalizer.to_device(device).normalize(inputs);

        let targets: Vec<Tensor<B, 1>> = dataset
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.y], device))
            .collect();

        let len = targets.len();
        let y = Tensor::cat(targets, 0);
        let x_test = Tensor::<B, 1>::from_floats([0.9026, 1.0], device).unsqueeze().to_device(device);
        let x_test = normalizer.to_device(device).normalize(x_test);
        Self{x, y, x_test,  len}
    }
}