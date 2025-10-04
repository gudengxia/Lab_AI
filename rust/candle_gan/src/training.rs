use std::time:: Instant;

use candle_core::{Result, Tensor, DType, D};
use candle_nn::{Optimizer};
use candle_nn as nn;
use super::{discriminator::Discriminator, generator::Generator};
use candle_datasets::vision::Dataset;
use super::dev;
use rand::{rng, seq::SliceRandom};
pub struct Learner{
    pub epochs: usize,
    pub batch_size: usize,
}

impl Learner{
    pub fn new(epochs: usize, batch_size: usize) -> Self{
        Learner{epochs, batch_size}
    }

    pub fn real_loss(&self, disc_pred: &Tensor)->Result<Tensor>{
        //let criterion = nn::loss::BCEWithLogitsLoss();
        // no BCEWithLogitsLoss in candle
        // replace it by binary_cross_entroy loss with a logsigmoid activation
        let ground_truth = disc_pred.ones_like()?;
        let logits = nn::ops::sigmoid(disc_pred)?;
        let r =  nn::loss::binary_cross_entropy_with_logit(&logits, &ground_truth);
        return r;
    }

    pub fn fake_loss(&self, disc_pred: &Tensor)->Result<Tensor>{
        //let criterion = nn::loss::BCEWithLogitsLoss();
        // no BCEWithLogitsLoss in candle
        // replace it by binary_cross_entroy loss with a logsigmoid activation
        let ground_truth = disc_pred.zeros_like()?;
        let logits = nn::ops::sigmoid(disc_pred)?;
        let r =  nn::loss::binary_cross_entropy_with_logit(&logits, &ground_truth);
        return r;
    }

    pub fn fit(&self, d:  &Discriminator, g: &Generator, m: &Dataset)->Result<()>{
        let train_images = m.train_images.to_device(&dev)?;
        let n= train_images.dim(0)?;
        let n_batches = n / self.batch_size;
        let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        let mut d_opt = d.configure_optimizers()?;
        let mut g_opt = g.configure_optimizers()?;
        let tensor_half = Tensor::from_vec([0.5].to_vec(), (), &dev)?;
        //important debug 20251004 by fzhang: the tensor's shape should be (), not 0 or 1
        let start_time = Instant::now();
        for echo in 1..(self.epochs + 1){
            let mut total_d_loss = Tensor::from_vec([0.0].to_vec(), (), &dev)?;
            let mut total_g_loss = Tensor::from_vec([0.0].to_vec(), (), &dev)?;
            
            batch_idxs.shuffle(&mut rng()); 

            // for each batch
            for batch_idx in batch_idxs.iter(){
                let batch_imgs = train_images.narrow(0, batch_idx * self.batch_size, self.batch_size)?;
                let mut nosie = Tensor::randn(0., 1., (self.batch_size, 64, 1, 1), &dev)?;
                let mut fake_img = g.forward(&nosie)?;
                
                //1. find loss and update weights for D
                let d_pred_fake = d.forward(&fake_img)?;
                let d_fake_loss = self.fake_loss(&d_pred_fake)?;
                let x_batch = batch_imgs.reshape((self.batch_size, 1, 28, 28))?.to_dtype(DType::F64)?;
                let d_pred_real = d.forward(&x_batch)?;
                let d_real_loss = self.real_loss(&d_pred_real)?;
                let d_loss = (d_fake_loss + d_real_loss)?.mul(&tensor_half)?;
                d_opt.backward_step(&d_loss)?;
                // let step_loss = d_loss.to_scalar::<f64>()?;
                // Debug 20251003 by fzhang: cannot use to_scalar or to_vec0
                // let step_loss = step_loss[0];
                total_d_loss = (total_d_loss + d_loss)?;
                //2. find loss and update weights for G
                nosie = Tensor::randn(0., 1., (self.batch_size, 64, 1, 1), &dev)?;
                fake_img = g.forward(&nosie)?;
                let d_pred = d.forward(&fake_img)?;
                let g_loss = self.real_loss(&d_pred)?;
                g_opt.backward_step(&g_loss)?;
                // update the model in the inner loop when finishing training each batch
                total_g_loss = (total_g_loss +  g_loss)?;
            }

            let avg_d_loss = total_d_loss.to_scalar::<f32>()? / n as f32;
            let avg_g_loss = total_g_loss.to_vec0::<f32>()? / n as f32;
            let elapsed_time = start_time.elapsed();
            println!("Epoch {}: d_loss = {}, g_loss = {}, consuming time {} s", echo, avg_d_loss, avg_g_loss, elapsed_time.as_secs_f64());    
        }
        Ok(())
    }
}