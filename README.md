# The Project
- Use CLIP and Vmamba to distillate a model
- Evaluate MAE Student vs Standard Student
- Use different masking in the Teacher and Student models to improve generalization

## Project Structure

```
StudentMAE
├── neural_network
│   └── base_model.py
├── data
├── utils
│   ├── logger.py
│   └── metrics.py
├── config
│   └── constants.py
├── evaluate_student.py
├── evaluate_student_mae.py
└── README.md
```

## Components

- **neural_network/base_classification_model.py**: Contains the `BaseClassificationModel` class, which serves as a foundation for building classification models. It includes methods for defining the model architecture, training steps, validation steps, and evaluation metrics.

- **data/**: This folder is designated for storing datasets. You can organize your datasets in subfolders or files as needed.

- **utils/logger.py**: Implements the `Logger` class for logging training progress, validation results, and other relevant information during model training and evaluation.

- **utils/metrics.py**: Provides functions for calculating evaluation metrics such as accuracy, precision, recall, and F1 score to assess model performance.

- **config/constants.py**: Defines constants used throughout the project, including learning rates, batch sizes, and other hyperparameters for consistency and ease of configuration.

- **evaluate_student.py**: A script for evaluating a student model without a masked autoencoder. It loads the model, processes the data, and computes evaluation metrics.

- **evaluate_student_mae.py**: A script for evaluating a student model with a masked autoencoder, incorporating the necessary functionality for evaluation.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd studentMAE
   ```

2. Install the required packages:

   Download CUDA 12.9 [here](https://developer.nvidia.com/cuda-12-9-0-download-archive)
   ```
   python -m venv .venv
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
   ```

3. Prepare your dataset and place it in the `data/` directory.

4. Modify the constants in `config/constants.py` as needed for your experiments.

5. Run the evaluation scripts:
   - Without masked autoencoder:
     ```
     python evaluate_student.py
     ```
   - With masked autoencoder:
     ```
     python evaluate_student_mae.py
     ```