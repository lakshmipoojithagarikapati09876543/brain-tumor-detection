# Brain Tumor Detection with EfficientNetB0 and Explainability

- Dataset: MRI brain images (tumor vs no tumor)
- Model: Transfer learning with EfficientNetB0
- Explainability: Grad-CAM visualizations to highlight important image regions
- Accuracy: ~90% (depends on dataset)
- Usage: Train on MRI scans, predict tumor presence, visualize heatmaps

## How to run

1. Prepare dataset folders: `data/train/tumor`, `data/train/no_tumor`, etc.
2. Install requirements: `pip install tensorflow matplotlib shap`
3. Run `python main.py` to train and save the model

## What I learned

- Transfer learning for medical imaging
- Model interpretability with Grad-CAM
- Image data augmentation
