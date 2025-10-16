#import datasets
#dataset = datasets.load_dataset("gvlassis/california_housing")
import pandas as pd
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/gvlassis/california_housing/" + splits["train"])