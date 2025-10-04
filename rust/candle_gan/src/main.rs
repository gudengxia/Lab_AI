use std::path::Path;
use candle_gan::training::Learner;
use candle_gan::discriminator::Discriminator;
use candle_gan::generator::Generator;
use candle_gan::gz_decoder;
use candle_datasets::vision::mnist;
use tracing::info;
fn main() -> candle_core::Result<()>{
    //let path = Path::new(r"D:\work\Lab_AI\gan\MNIST\MNIST\");
    //println!("path:{:?}", path);
    //let m = mnist::load_dir(path)?;
    //let m = mnist::load()?;
    tracing_subscriber::fmt::init();
    let dataset_dir = "../../gan/MNIST/MNIST/raw";
    gz_decoder::decompress_dataset(dataset_dir);

    let m = candle_datasets::vision::mnist::load_dir(dataset_dir)?;

    info!("train-images: {:?}",m.train_images.shape());
    info!("train-labels: {:?}", m.train_labels.shape());
    info!("test-images: {:?}", m.test_images.shape());
    info!("test-labels: {:?}", m.test_labels.shape());

    let learner = Learner{epochs: 5, batch_size: 128};
    let d = Discriminator::new()?;
    let g = Generator::new()?;
    learner.fit(&d, &g, &m)?;

    Ok(())
}
