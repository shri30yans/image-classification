# Image Classification with PyTorch
## Overview
This project aims to accurately classify images across various categories following learning from diverse datasets using the PyTorch and Torchvision

## Features
- Train and test the model to classify images into distinct categories.
- Leverages PyTorch, TorchVision for efficient machine learning techniques.
- Flexible model architecture and training parameters for experimentation.

## Usage
1. Install Dependecies
```
pip install -r requirements.txt
```

2. **Prepare Data:**
  Place your training and testing images in their respective folders within the `images` directory.
```
images/
├── training/
│ ├── category_1/
│ ├── category_2/
│ └── ...
└── testing/
├── category_1/
├── category_2/
└── ...
```
3. **Train the Model:**
   Run the training script.
```bash
python main.py
```

4. **Test the Model:**
Run the testing script to classify new images.
```bash
python testing_model.py
```

## Additional Notes
Ensure that your images are properly labeled and organized into separate folders representing different categories.
Experiment with different model architectures, hyperparameters, and training strategies to optimize classification performance.

## Acknowledgements
This project was developed as part of my exploration and experimenting into image classification techniques and was inspired by various tutorials, documentation, and online resources.
