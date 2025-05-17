# COS720-Final-Project-2025

## Folder Structure
- API - Contains the main code for the phishing detection API and classification model.
- Experiment - Contains the code for the training and evaluation of the classification model.
  - archive - Contains the original dataset used for training and evaluation.

## Running the Code
### Prerequisites
- Python 3.12.7
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the API
```bash
cd API
# Run the API
python -u api_v2.py
```

### Run the Experiment
This code is a jupyter notebook that trains and evaluates a phishing detection model using a dataset of phishing and legitimate URLs. The notebook includes data preprocessing, feature extraction, model training, and evaluation steps.