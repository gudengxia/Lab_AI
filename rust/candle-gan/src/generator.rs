use candle_core::{ DType, Result, Tensor};
use candle_nn as nn;
use nn::{Module, VarMap, VarBuilder};
use nn::{optim::AdamW, ParamsAdamW, Optimizer};
use super::dev;

#[derive(Clone)]
struct GBlock{
    conv_t: nn::ConvTranspose2d,
    bn: nn::BatchNorm,
    relu: nn::activation::Activation
}

impl GBlock{
    fn new(block_index: usize, in_channel: usize, out_channel: usize, kernel_size: usize, stride: usize, vars: &mut VarMap)
    ->Result<Self>{
        let vb = VarBuilder::from_varmap(&vars, candle_core::DType::F64, &dev);
        //let mut conv2d_config: nn::conv::Conv2dConfig = Default::default();
        //conv2d_config.stride = stride;
        // out_dim = (in_dim-kernel_size)/stride+1 
        //(bs, 1, 28, 28)
        let mut name = format!("cov_t_lay{}", block_index);
        //let conv = nn::ConvTranspose2d(in_channel, out_channel, kernel_size, conv2d_config, vb.pp(name))?;
        let conv_t_config = nn::ConvTranspose2dConfig{stride: stride, ..Default::default()};
        let conv_t = nn::conv_transpose2d(in_channel, out_channel, kernel_size, conv_t_config, vb.pp(name))?; 
        name = format!("bn_lay{}", block_index);
        let bn_config: nn::BatchNormConfig = Default::default();
        let bn = nn::batch_norm(out_channel, bn_config, vb.pp(name))?;
        let relu = nn::activation::Activation::Relu;
        Ok(Self{conv_t, bn, relu})
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor>{
        let x1 = self.conv_t.forward(x)?;
        let x2 = x1.apply_t(&(self.bn), false)?;
        //let x3 = nn::ops::leaky_relu(&x2, self.lr)?;
        let x3 = self.relu.forward(&x2)?;
        return Ok(x3);
    }
} 
pub struct Generator{
    params: VarMap,
    block1: GBlock,
    block2: GBlock,
    block3: GBlock,
    conv_t: nn::ConvTranspose2d,
    tanh: nn::Activation
}

impl Generator{
    pub fn new()->Result<Self>{
        let mut vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F64, &dev);
        let noise_dim = 64;
        // input: (bs, nosie_dim)
        // reshape(bs, channel, height, width) -> (bs, noism_dim, 1, 1)
        let block1 = GBlock::new(1, noise_dim , 256, 3, 2, &mut vars)?;
        // (bs, 256, 3, 3)
        let block2 = GBlock::new(2, 256, 128, 4, 1, &mut vars)?;
        // (bs, 128, 6, 6)
        let block3 = GBlock::new(3, 128, 64, 3, 2, &mut vars)?;
        //(bs, 64, 13, 13)
        let conv_t_config = nn::ConvTranspose2dConfig{stride: 2, ..Default::default()};
        let conv_t = nn::conv_transpose2d(64, 1, 4, conv_t_config, vb.pp("conv_t_final"))?;
        //(bs, 1, 28, 28)
        let tanh = nn::Activation::GeluPytorchTanh;
        Ok(Self{params: vars, block1, block2, block3, conv_t, tanh})
    }


    pub fn forward(&self, image: &Tensor)->Result<Tensor>{
        let x1 = self.block1.forward(image)?;
        let x2 = self.block2.forward(&x1)?;
        let x3 = self.block3.forward(&x2)?;
        let x4 = self.conv_t.forward(&x3)?;
        let x5 = self.tanh.forward(&x4)?;
        return Ok(x5);
    } 

    pub fn configure_optimizers(&self)->Result<nn::optim::AdamW>{
        let params_adam = ParamsAdamW{lr:0.0002, beta1: 0.5, beta2: 0.99, ..Default::default()};
        let optimizer = AdamW::new(self.params.all_vars(), params_adam);
        return optimizer;
    }
}