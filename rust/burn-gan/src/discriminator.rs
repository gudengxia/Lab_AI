use burn::nn::{BatchNormConfig, LeakyRelu, LeakyReluConfig};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm,
        Linear, LinearConfig,
    },
    prelude::*,
};
use crate::{bce_with_binary_entroy, BATCH_SIZE};

use super::rand_nosie;
#[derive(Module, Debug)]
pub struct DBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    relu: LeakyRelu,
}

#[derive(Config, Debug)]
pub struct DBlockConfig{
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize
}

impl DBlockConfig{
    fn get_disc_block<B: Backend>(in_channels :usize, out_channels :usize, kernel_size: usize, stride: usize, device: &B::Device) -> DBlock<B>{
        let mut conv_config = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size]);
        conv_config = conv_config.with_stride([stride, stride]);
        let conv: Conv2d<B> = conv_config.init(device);
        let bn: BatchNorm<B, 2> =  BatchNormConfig::new(out_channels).init(device);
        let relu_config = LeakyReluConfig::new().with_negative_slope(0.02);
        let relu = relu_config.init();
        DBlock{conv, bn, relu}
    }
}


impl<B: Backend> DBlock<B>{
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4>{
        // Create a channel at the second dimension.
        //let x = images.reshape([batch_size, 1, height, width]);
        let x = self.conv.forward(input); // [batch_size, 8, _, _]
        let x = self.bn.forward(x);
        let x = self.relu.forward(x);
        return x;
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    block1: DBlock<B>,
    block2: DBlock<B>,
    block3: DBlock<B>,
    linear: Linear<B>
}


#[derive(Config, Debug)]
pub struct DiscriminatorConfig{
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl<B: Backend> Discriminator<B> {
    pub fn new(dev: &B::Device)->Self{
        let block1 = DBlockConfig::get_disc_block(1, 16, 3, 2, dev);
        let block2 = DBlockConfig::get_disc_block(16, 32, 5, 2, dev);
        let block3 = DBlockConfig::get_disc_block(32, 64, 5, 2, dev);
        let linear = LinearConfig::new(64, 1).init(dev);
        Discriminator { block1, block2, block3, linear}
    }

    pub fn forward(&self, input: &Tensor<B, 4>) -> Tensor<B, 2> {   // input: [batchsize, 1, 28, 28]
        let x :Tensor<B, 4> = self.block1.forward(input.clone());     // [batch_size, 16, 13, 13]
        let x :Tensor<B, 4> = self.block2.forward(x);          // [batch_size, 16, 5, 5]
        let x :Tensor<B, 4> = self.block3.forward(x);          // [batch_size, 64, 1, 1]
        let x :Tensor<B, 2> = x.flatten(1, 3);    // [batch_size, 64]
        let x :Tensor<B, 2> = self.linear.forward(x);          //[batch_size, 1] 
        x                                                           
    }

    pub fn forward_step_for_classification(&self, real_images: &Tensor<B, 4>) -> ClassificationOutput<B> {
        let dev = &real_images.device();
        let fake_images = rand_nosie(dev);
        let y_real = self.forward(&real_images);
        let y_fake = self.forward(&fake_images);
        let ones = y_real.ones_like();
        let zeros = y_fake.zeros_like();
        let dev = real_images.device();
        let loss_real = bce_with_binary_entroy(&y_real, &ones, &dev);
        let loss_fake = bce_with_binary_entroy(&y_fake, &zeros, &dev);

        let d_loss = (loss_real + loss_fake) / 2.0;
        //TrainOutput<T0>{grads: GradientsParams, item: T0}
        //RegressionOutput{loss, output, targets}
        let y_hat = Tensor::cat([y_real, y_fake].to_vec(), 0);
        let y = Tensor::cat([ones, zeros].to_vec(), 0);
        let item = ClassificationOutput::new(d_loss, y_hat, y.reshape([BATCH_SIZE*2]).int());
        item        
    }
}
//implement the TrainStep Feature for model
impl<B: AutodiffBackend> TrainStep<Tensor<B, 4>, ClassificationOutput<B>> for Discriminator<B> {
    fn step(&self, images: Tensor<B, 4>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_step_for_classification(&images);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Tensor<B, 4>, ClassificationOutput<B>> for Discriminator<B> {
    fn step(&self, images: Tensor<B, 4>) -> ClassificationOutput<B> {
        let item = self.forward_step_for_classification(&images);
        item
    }
}