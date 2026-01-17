# Automated Material Recognition

## Problem Setup
- Supervised single-label image classification
- 6 classes: Cardboard, Glass, Metal, Paper, Plastic, General Waste
- Address class imbalance where possible

## Data Preparation
- Resize images to model input size (Probably 224×224 as good balance of performance & speed)
- Normalise using ImageNet mean and standard deviation
- Data augmentation:
  - Random crops
  - Horizontal flips
  - Brightness / contrast adjustments

## Model Architecture
- Use a CNN backbone pretrained on ImageNet
- Recommended architectures:
  - ResNet-50 / ResNet-101
  - EfficientNet (B0–B3)
- Replace final layer with a 6-unit dense layer + softmax

## Transfer Learning
- Initialise with pretrained ImageNet weights
- Freeze backbone initially; train classifier head only
- Gradually unfreeze top layers and fine-tune with low learning rate

## Training
- Loss: categorical cross-entropy
- Optimiser: Adam bc it's the goat
- Apply class weights or focal loss if imbalance remains
- Use learning-rate scheduling and early stopping to set epochs high asf and leave to train

## Evaluation
- Metrics:
  - Overall accuracy
  - Per-class precision and recall
- Analyse confusion matrix (e.g. Paper vs Cardboard)
- Perform qualitative error analysis on improppperly classified images

## Possibilities
- Higher input res for fine-grained classes (May help class confusion e.g. paper vs cardboard)
- Model ensembling for improved accuracy (If I can be bothered/have time)
- Confidence thresholds for uncertain predictions (Will need to figure out)
- Detection or segmentation if multiple items per image (Big effort EDA first using yolo)
