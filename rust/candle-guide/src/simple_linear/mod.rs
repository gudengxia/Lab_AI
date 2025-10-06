
// ANCHOR: book_training_simplified1
use candle_core::Device;

const device: Device = Device::Cpu;

pub mod dataset;
pub mod model;
pub mod training;
pub mod a_linear;