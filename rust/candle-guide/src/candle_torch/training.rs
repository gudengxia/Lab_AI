use candle_core::{DType, Result, D};
use candle_nn::{ops, Optimizer, VarBuilder, VarMap};
use crate::candle_torch::{dataset::Dataset, model:: {MultiLevelPerceptron,ExPerceptron}};
use super::*;

pub struct Trainer{
}

impl Trainer {
    pub fn fit(&self, model: &MultiLevelPerceptron, data: Dataset)->Result<MultiLevelPerceptron>{
        let mut sgd = model.configure_optimizers()?;
        let x_train = data.x_train.to_device(&device)?;
        let y_train = data.y_train.to_device(&device)?;
        let x_test = data.x_test.to_device(&device)?;
        let y_test = data.y_test.to_device(&device)?;
        let mut final_accuracy: f32;
        for epoch in 1..(EPOCHS + 1){
            let logits = model.forward(&x_train)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            //let loss = loss::nll(&log_sm, &data.y_train)?;
            let loss = model.loss(&log_sm, &y_train)?;
            sgd.backward_step(&loss)?;
            
            let test_logits = model.forward(&x_test)?;

            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&y_test)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            
            let test_accuracy = sum_ok / y_test.dims1()? as f32;
            final_accuracy = 100. * test_accuracy;
            println!("Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                    loss.to_scalar::<f32>()?,
                    final_accuracy
            );
            if final_accuracy == 100.0 {
                break;
            }
        }
        Ok((*model).clone())
    }


    pub fn train(&self, m: Dataset) -> anyhow::Result<ExPerceptron> {
        let train_results = m.y_train.to_device(&device)?;
        let train_votes = m.x_train.to_device(&device)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = ExPerceptron::new(vs)?;
        
        let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
        //let mut sgd = candle_nn::SGD::new(model.params.all_vars(), LEARNING_RATE)?;
        //let mut sgd = model.configure_optimizers()?;
        let test_votes = m.x_test.to_device(&device)?;
        let test_results = m.y_test.to_device(&device)?;
        let mut final_accuracy: f32 = 0.0;
        for epoch in 1..EPOCHS + 1 {
            let logits = model.forward(&train_votes)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = candle_nn::loss::nll(&log_sm, &train_results)?;
            sgd.backward_step(&loss)?;

            let test_logits = model.forward(&test_votes)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_results)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_results.dims1()? as f32;
            final_accuracy = 100. * test_accuracy;
            println!("Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                    loss.to_scalar::<f32>()?,
                    final_accuracy
            );
            if final_accuracy == 100.0 {
                break;
            }
        }
        if final_accuracy < 100.0 {
            Err(anyhow::Error::msg("The model is not trained well enough."))
        } else {
            Ok(model)
        }
    }
}