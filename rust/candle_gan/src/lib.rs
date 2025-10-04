//candel-gan
pub mod discriminator;
pub mod generator;
pub mod training;
pub mod gz_decoder;
const dev: candle_core::Device =  candle_core::Device::Cpu;