use burn::{
    module::Module,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::{backend::Backend, Tensor}
};



use crate::a_simple_linear::{
    model::{RegressionModelConfig, RegressionModelRecord},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let record: RegressionModelRecord<B> = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = RegressionModelConfig::new()
        .init(&device)
        .load_record(record);
    
    let x = Tensor::<B,2>::from_data([[0.9026f32, 1.0f32],[1.0f32, 1.0f32]], &device);

    let y_hat = model.forward(x);

    let predicted = y_hat.into_data().iter::<f32>()
        .collect::<Vec<_>>();

    // Print a single numeric value as an example
    // Print a single numeric value as an example
    println!("Predicted y is {:?}", predicted);
}