use candle_core::{DType, Result, Tensor};
use candle_nn as nn;
use nn::{Module, VarMap, VarBuilder};
use nn::{optim::AdamW, ParamsAdamW, Optimizer};
use super::dev;

#[derive(Clone)]
struct DBlock{
    conv: nn::Conv2d,
    bn: nn::BatchNorm,
    relu: nn::activation::Activation
}

impl DBlock{
    fn new(block_index: usize, in_channel: usize, out_channel: usize, kernel_size: usize, stride: usize, lr: f64, vars: &mut VarMap)
    ->Result<Self>{
        let vb = VarBuilder::from_varmap(&vars,DType::F64, &dev);
        let mut conv2d_config: nn::conv::Conv2dConfig = Default::default();
        conv2d_config.stride = stride;
        // out_dim = (in_dim-kernel_size)/stride+1 
        //(bs, 1, 28, 28)
        let mut name = format!("cov_lay{}", block_index);
        let conv = nn::conv2d(in_channel, out_channel, kernel_size, conv2d_config, vb.pp(name))?;
        name = format!("bn_lay{}", block_index);
        let bn_config: nn::BatchNormConfig = Default::default();
        let bn = nn::batch_norm(out_channel, bn_config, vb.pp(name))?;
        let relu = nn::activation::Activation::LeakyRelu(lr);
        Ok(Self{conv, bn, relu})
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor>{
        let x1 = self.conv.forward(x)?;
        let x2 = x1.apply_t(&(self.bn), false)?;
        //let x3 = nn::ops::leaky_relu(&x2, self.lr)?;
        let x3 = self.relu.forward(&x2)?;
        return Ok(x3);
    }
} 
pub struct Discriminator{
    params: VarMap,
    block1: DBlock,
    block2: DBlock,
    block3: DBlock,
    linear: nn::Linear
}

impl Discriminator{
    pub fn new()->Result<Self>{
        let lr = 0.2;
        let mut vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F64, &dev);
        let block1 = DBlock::new(1, 1, 16, 3, 2, lr, &mut vars)?;
        let block2 = DBlock::new(2, 16, 32, 5, 2, lr, &mut vars)?;
        let block3 = DBlock::new(3, 32, 64, 5, 2, lr, &mut vars)?;
        let linear = nn::linear(64, 1, vb.pp("ln"))?;
        Ok(Self{params: vars, block1, block2, block3, linear})
        /*let mut conv2d_config: nn::conv::Conv2dConfig = Default::default();
        conv2d_config.stride = 2;
        // out_dim = (in_dim-kernel_size)/stride+1 
        //(bs, 1, 28, 28)
        let conv1 = nn::conv2d(1, 16, 3, conv2d_config, vb.pp("cv1"))?;
        //(bs, 16, 13, 13)
        let conv2 = nn::conv2d(16, 32, 5, conv2d_config, vb.pp("cv2"))?;
        //(bs, 32, 5, 5)
        let conv3 = nn::conv2d(1, 16, 3, conv2d_config, vb.pp("cv1"))?;
        //(bs, 64, 1, 1)
        
        let bn_config: nn::BatchNormConfig = Default::default();
        let bn1 = nn::batch_norm(64, bn_config, vb.pp("bn1"))?;
        let bn2 = nn::batch_norm(64, bn_config, vb.pp("bn2"))?;
        let bn3 = nn::batch_norm(64, bn_config, vb.pp("bn3"))?;*/
    }


    pub fn forward(&self, image: &Tensor)->Result<Tensor>{
        let x1 = self.block1.forward(image)?;
        let x2 = self.block2.forward(&x1)?;
        let x3 = self.block3.forward(&x2)?;
        let x4 = x3.flatten(1, 3)?; //debug 20251003: flatten the last three dimensions
        let x5 = self.linear.forward(&x4)?;
        return Ok(x5);
    } 

    pub fn configure_optimizers(&self)->Result<nn::optim::AdamW>{
        let params_adam = ParamsAdamW{lr:0.0002, beta1: 0.5, beta2: 0.99, ..Default::default()};
        //let optimizer = nn::optim::Optimizer::AdamW::new(self.paraAdams.all_vars(), params_adam);
        let optimizer = AdamW::new(self.params.all_vars(), params_adam);
        return optimizer;
    }
}