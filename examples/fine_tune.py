"""This example demonstrates how to use the `fine_tune` and `sample` methods of VectGan."""

import os

import pandas as pd

from vect_gan.synthesizers.vectgan import VectGan

vect_gan = VectGan()

# Load some sample data
current_directory = os.path.dirname(os.path.abspath(__file__))
data = pd.read_parquet(f"{current_directory}/../data/sample_data.parquet")

model = vect_gan.fine_tune(new_data=data, epochs=10, batch_size=32, pac=8, verbose=True)

# Sample 10 synthetic data points from the fine-tuned model
sampled_data = model.sample(10)
print(sampled_data)

# Save the fine-tuned model
model.save("fine_tuned_model.pt")

# Example loading of the model
vect_gan = vect_gan.load("fine_tuned_model.pt")
