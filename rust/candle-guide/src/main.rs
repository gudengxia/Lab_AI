
use candle_core::Result;
use candle_guide::rtorch::multilinear::{model, dataset, training};
use candle_guide::rtorch::multilinear::example::simplified;

#[tokio::main]
async fn main()->Result<()>{
    //simplified();

    let data = dataset::Dataset::new()?;
    let model = model::MultiLevelPerceptron::new()?;
    let trainer = training::Trainer::new();  
    trainer.fit(&model, data)?;
    //trainer.train(data);
    model.validate()?;
    Ok(())
}
/*async fn main(){
    const device: Device = Device::Cpu;
    //simplified();
    let w = Tensor::from_vec([2f32, -3.4f32].to_vec(), (2, 1), &device).expect("error!");
    
    let b = Tensor::from_vec([4.2f32].to_vec(), (1, 1), &device).expect("error!");
    
    let noise = 0.01f32;
    
    let dataset = Dataset::new(&w, &b, noise, 10, 2).expect("error in dataset.");
    
    println!("{:?}, {:?}", dataset.v_x, dataset.v_y);
    //println!("{:?}", y);
}*/

/*struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::Cpu;

    // This has changed (784, 100) -> (100, 784) !100->3, 784->5, 10->2
    let weight = Tensor::randn(0f32, 1.0, (3, 5), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (3, ), &device)?;
    println!("first:{weight:?}, {bias:?}");
    let first = Linear::new(weight, Some(bias));
    let weight = Tensor::randn(0f32, 1.0, (2, 3), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (2, ), &device)?;
    println!("second:{weight:?}, {bias:?}");
    let second = Linear::new(weight, Some(bias));
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 5), &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}*/