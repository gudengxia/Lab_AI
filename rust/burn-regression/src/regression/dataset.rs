use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset},
    },
    prelude::*,
};

pub const NUM_FEATURES: usize = 8;

// Pre-computed statistics for the housing dataset features
const FEATURES_MIN: [f32; NUM_FEATURES] = [0.4999, 1., 0.8461, 0.375, 3., 0.6923, 32.54, -124.35];
const FEATURES_MAX: [f32; NUM_FEATURES] = [
    15., 52., 141.9091, 34.0667, 35682., 1243.3333, 41.95, -114.31,
];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HousingDistrictItem {
    /// Median income
    #[serde(rename = "MedInc")]
    pub median_income: f32,

    /// Median house age
    #[serde(rename = "HouseAge")]
    pub house_age: f32,

    /// Average number of rooms per household
    #[serde(rename = "AveRooms")]
    pub avg_rooms: f32,

    /// Average number of bedrooms per household
    #[serde(rename = "AveBedrms")]
    pub avg_bedrooms: f32,

    /// Block group population
    #[serde(rename = "Population")]
    pub population: f32,

    /// Average number of household members
    #[serde(rename = "AveOccup")]
    pub avg_occupancy: f32,

    /// Block group latitude
    #[serde(rename = "Latitude")]
    pub latitude: f32,

    /// Block group longitude
    #[serde(rename = "Longitude")]
    pub longitude: f32,

    /// Median house value (in 100 000$)
    #[serde(rename = "MedHouseVal")]
    pub median_house_value: f32,
}

pub struct HousingDataset {
    dataset: SqliteDataset<HousingDistrictItem>,
}

impl Dataset<HousingDistrictItem> for HousingDataset {
    fn get(&self, index: usize) -> Option<HousingDistrictItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl HousingDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn validation() -> Self {
        Self::new("validation")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<HousingDistrictItem> =
            HuggingfaceDatasetLoader::new("gvlassis/california_housing")
                .dataset(split)
                .unwrap();

        Self { dataset }
    }
}

/// Normalizer for the housing dataset.
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

    /// Normalizes the input image according to the housing dataset min/max.
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
pub struct HousingBatcher<B: Backend> {
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct HousingBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> HousingBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            normalizer: Normalizer::new(&device, &FEATURES_MIN, &FEATURES_MAX),
        }
    }
}

impl<B: Backend> Batcher<B, HousingDistrictItem, HousingBatch<B>> for HousingBatcher<B> {
    fn batch(&self, items: Vec<HousingDistrictItem>, device: &B::Device) -> HousingBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.median_income,
                    item.house_age,
                    item.avg_rooms,
                    item.avg_bedrooms,
                    item.population,
                    item.avg_occupancy,
                    item.latitude,
                    item.longitude,
                ],
                device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.normalizer.to_device(device).normalize(inputs);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.median_house_value], device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        HousingBatch { inputs, targets }
    }
}


#[derive(Clone, Debug)]
pub struct HousingData<B: Backend> {
    pub x_train: Tensor<B, 2>,
    pub y_train: Tensor<B, 1>,
    pub x_test: Tensor<B, 2>,
    pub y_test: Tensor<B, 1>
}
impl<B: Backend> HousingData<B> {
    pub fn new(device: &B::Device) -> Self {
        let dataset_train = HousingDataset::train();
        let mut x_train_vec: Vec<Tensor<B, 2>> = Vec::new(); 
        let mut y_train_vec: Vec<Tensor<B, 1>> = Vec::new();

        for i in 0..dataset_train.len() {
            let item = dataset_train.get(i).unwrap();
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.median_income,
                    item.house_age,
                    item.avg_rooms,
                    item.avg_bedrooms,
                    item.population,
                    item.avg_occupancy,
                    item.latitude,
                    item.longitude,
                ],
                device,
            );

            x_train_vec.push(input_tensor.unsqueeze());
            y_train_vec.push(Tensor::<B, 1>::from_floats([item.median_house_value], device));
        }

        let x_train = Tensor::cat(x_train_vec, 0);
        let y_train = Tensor::cat(y_train_vec, 0);

        let normalizer = Normalizer::new(device, &FEATURES_MIN, &FEATURES_MAX);
        let x_train = normalizer.normalize(x_train);

        let dataset_test = HousingDataset::test();
        let mut x_test_vec: Vec<Tensor<B, 2>> = Vec::new(); 
        let mut y_test_vec: Vec<Tensor<B, 1>> = Vec::new();

        for i in 0..dataset_test.len() {
            let item = dataset_test.get(i).unwrap();
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.median_income,
                    item.house_age,
                    item.avg_rooms,
                    item.avg_bedrooms,
                    item.population,
                    item.avg_occupancy,
                    item.latitude,
                    item.longitude,
                ],
                device,
            );

            x_test_vec.push(input_tensor.unsqueeze());
            y_test_vec.push(Tensor::<B, 1>::from_floats([item.median_house_value], device));
        }

        let x_test = Tensor::cat(x_test_vec, 0);
        let y_test = Tensor::cat(y_test_vec, 0);
        let x_test = normalizer.normalize(x_test);

        Self {
            x_train,
            y_train,
            x_test,
            y_test
        }
    }
}