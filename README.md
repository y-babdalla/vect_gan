# VECT-GAN: A Variationally Encoded Generative Model for Overcoming Data Scarcity in Pharmaceutical Science

> **Authors**: Youssef Abdalla, Marrisa Taub, Priya Akkaru, Eleanor Hilton, Alexander Milanovic, Mine Orlu, Abdul W. Basit, Michael T Cook, Tapabrata Chakraborti, David Shorthouse  
> **Institutions**:
>  - Department of Pharmaceutics, UCL School of Pharmacy, University College London, London, UK  
>  - The Alan Turing Institute, London, UK  

VECT-GAN is a research-oriented Python package designed to generate synthetic tabular data, specifically focused on molecular data. Its functionality includes data transformation, conditional data sampling, and model fine-tuning. Although developed for generating molecular data, the principles can be easily adapted to other domains.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model from Scratch](#training-the-model-from-scratch)
  - [Fine-Tuning the Molecular Descriptor Model](#fine-tuning-the-molecular-descriptor-model)
  - [Sampling Synthetic Data](#sampling-synthetic-data)
  - [Saving and Loading Models](#saving-and-loading-models)
- [Contributing](#contributing)
- [Licence](#licence)
- [Contact](#contact)

---

## Installation

VECT-GAN requires Python 3.11 or later. (While it may run on earlier versions, it is officially tested on Python 3.11+.)

Install via pip:

```bash
pip install vect-gan
```

Alternatively, if you have the source code:

```bash
git clone https://github.com/y-babdalla/vect_gan.git
cd vect_gan
pip install .
```

Once installed, you can import the package in Python:

```python
import vect_gan
```

---

## Usage

### Training the Model from Scratch

Before fine-tuning or using a pre-trained VECT-GAN model, you can train a new model from scratch using the `.fit` method. This step is recommended if you have a novel dataset and wish to initialise the generative model from the ground up.

```python
import pandas as pd
from vect_gan.synthesizers.vectgan import VectGan

# Create an instance of VectGan
vect_gan = VectGan()

# Load your training data into a pandas DataFrame
training_data = pd.read_csv("path/to/your/training_data.csv")

# Train the model from scratch
trained_model = vect_gan.fit(
    data=training_data,
    epochs=200,
    batch_size=64,
    pac=8,
    verbose=True
)
```

- **data**: The training dataset (pandas DataFrame).  
- **epochs**: The number of training epochs.  
- **batch_size**: Training batch size.  
- **pac**: Pack size for training (grouping samples for improved stability). Please note that the `pac` size must be divisible by the batch size.
- **verbose**: Controls the verbosity of training logs.

### Fine-Tuning the Molecular Descriptor Model

Once you have a trained or pre-trained model, you may wish to *fine-tune* it on additional data to adapt the model’s learned representations to new or more specialised tasks. Fine-tuning generally uses fewer epochs than a full training routine.

> **Note**: Because VECT-GAN’s underlying `DataTransformer` and `DataSampler` are *not* serialisable, they are **not** included in the pre-trained model checkpoint. If you download or receive a pre-trained VECT-GAN model, you still need to run a brief fine-tuning step on a small amount of data. We provide some synthetic data examples in the `data/` folder that you can use for this purpose.

```python
import pandas as pd
from vect_gan.synthesizers.vectgan import VectGan

# Create an instance of VectGan
vect_gan = VectGan()

# Load data for fine-tuning
new_data = pd.read_csv("path/to/your/data.csv")

# Fine-tune the model
model = vect_gan.fine_tune(
    new_data=new_data,
    epochs=10,
    batch_size=32,
    pac=8,
    verbose=True
)
```

- **new_data**: The dataset on which to perform fine-tuning (pandas DataFrame).  
- **epochs**: The number of training epochs.  
- **batch_size**: Training batch size.  
- **pac**: Pack size for training (grouping samples for improved stability). Please note that the `pac` size must be divisible by the batch size.  
- **verbose**: Controls the verbosity of logs.

### Sampling Synthetic Data

Once you have trained or fine-tuned a model, you can generate synthetic data:

```python
# Sample 10 rows of synthetic data
sampled_data = model.sample(10)

print("Sampled Data:")
print(sampled_data)
```

### Saving and Loading Models

VECT-GAN uses `torch.save` and `torch.load` under the hood to handle model checkpoints:

```python
# Save the fine-tuned model
model.save("fine_tuned_model.pt")

# Load the model from disk
vect_gan = vect_gan.load("fine_tuned_model.pt")
```

The above commands store or retrieve the learned model weights. However, as noted, the data transformation objects are **not** serialised. This means the loaded model will require an external transformer and data sampler to operate fully. If you want to run the model immediately upon loading (e.g. to generate new samples), you will need to either:

- Fine-tune the loaded model on a small dataset (as described above), or
- Re-initialise the same type of `DataTransformer` and `DataSampler` that were used originally.

---

## Contributing

We welcome contributions in the form of **bug reports**, **feature requests**, or **pull requests**. If you wish to contribute code, kindly follow these steps:

1. **Fork** this repository.  
2. **Create** a new branch from `main`.  
3. **Implement** or fix the feature/bug.  
4. **Submit** a pull request, describing your changes in detail.

Please ensure your code follows Python 3 best practices, PEP8 style conventions, and includes comprehensive docstrings.

---

## Licence

VECT-GAN is distributed under the **GNU GENERAL PUBLIC LICENSE**. Please refer to the [LICENSE](LICENSE) file in the repository for detailed information.

---

## Contact

For questions, comments, or feedback regarding VECT-GAN, please contact:

- **Lead Author**: Youssef Abdalla (youssef.abdalla.16@ucl.ac.uk)
- **GitHub Issues**: [https://github.com/y-babdalla/vect_gan/issues](https://github.com/y-babdalla/vect_gan/issues)

Please note that the data used to train the models is private. For access to the training and evaluation data, please contact the authors.

If you use VECT-GAN in academic work, kindly cite our research paper. This project is continually evolving, and we appreciate any input that could help improve it.
