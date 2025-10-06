use burn::nn::{BatchNormConfig, Relu};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
    nn::{
        conv::{ConvTranspose2d, ConvTranspose2dConfig},
        BatchNorm,
        Tanh
    },
    prelude::*,
};
use crate::{bce_with_binary_entroy, BATCH_SIZE};
use crate::discriminator::Discriminator;

#[derive(Module, Debug)]
pub struct GBlock<B: Backend> {
    conv_t: ConvTranspose2d<B>,
    bn: BatchNorm<B, 2>,
    relu: Relu
}


fn get_gen_block<B: Backend>(in_channels :usize, out_channels :usize, kernel_size: usize, stride: usize, device: &B::Device) -> GBlock<B>{
    let mut conv_t_config = ConvTranspose2dConfig::new([in_channels, out_channels],[kernel_size, kernel_size]);
    conv_t_config = conv_t_config.with_stride([stride, stride]);
    let conv_t: ConvTranspose2d<B> = conv_t_config.init(device);
    let bn: BatchNorm<B, 2> =  BatchNormConfig::new(out_channels).init(device);
    let relu = Relu::new();
    GBlock{conv_t, bn, relu}
}


impl<B: Backend> GBlock<B>{
    pub fn forward(&self, input: &Tensor<B, 4>) -> Tensor<B, 4>{
        // Create a channel at the second dimension.
        //let x = images.reshape([batch_size, 1, height, width]);
        let x = self.conv_t.forward(input.clone()); // [batch_size, 8, _, _]
        let x = self.bn.forward(x);
        let x = self.relu.forward(x);
        return x;
    }
}

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    block1: GBlock<B>,
    block2: GBlock<B>,
    block3: GBlock<B>,
    conv_t: ConvTranspose2d<B>,
    tanh: Tanh,
    disc: Discriminator<B>
}



impl<B: Backend> Generator<B> {
    pub fn new(noise_dim: usize, disc: Discriminator<B>, dev: &B::Device)->Self{
        let block1 = get_gen_block(noise_dim, 256, 3, 2, dev);
        let block2 = get_gen_block(256, 128, 4, 1, dev);
        let block3 = get_gen_block(128, 64, 3, 2, dev);
        let conv_t = ConvTranspose2dConfig::new([64, 1], [4, 4]).with_stride([2, 2]).init(dev);
        let tanh = Tanh::new();
        Generator{ block1, block2, block3, conv_t, tanh, disc}
    }

    pub fn forward(&self, batch: &Tensor<B, 4>) -> Tensor<B, 2> {// input: [bs, nosie_dim, 1, 1]
        let x = self.block1.forward(batch);            // [batch_size, 256, 3, 3]
        let x = self.block2.forward(&x);         // [batch_size, 128, 6, 6]
        let x = self.block3.forward(&x);         // [batch_size, 64, 13, 13]
        let x = self.conv_t.forward(x);         // [batch_size, 1, 28, 28]
        let x = self.tanh.forward(x);           //[batch_size, 1, 28, 28] 
        let x = self.disc.forward(&x);
        x                                                         
    }

    pub fn forward_step(&self, batch: &Tensor<B, 4>, dev: &B::Device) -> ClassificationOutput<B> {
        let y_fake = self.disc.forward(&batch);
        let zeros = y_fake.zeros_like();
        let loss = bce_with_binary_entroy(&y_fake, &zeros, &dev);
        let item = ClassificationOutput::new(loss, y_fake, zeros.reshape([BATCH_SIZE]).int());
        item     
    }
}

/* Cannot find a way to pass the Discriminator object to step()
impl<B: AutodiffBackend> TrainStep<Tensor<B, 4>, ClassificationOutput<B>> for Generator<B> {
    fn step(&self, batch: Tensor<B, 4>) -> TrainOutput<ClassificationOutput<B>> {
        let dev = batch.device();
        let item = self.forward_step(&batch, &dev);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Tensor<B, 4>, ClassificationOutput<B>> for Generator<B> {
    fn step(&self, batch: Tensor<B, 4>) -> ClassificationOutput<B> {
        let dev = batch.device();
        self.forward_step(&batch, &dev)
    }
}*/