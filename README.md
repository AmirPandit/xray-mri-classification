```markdown
# 🧠 X-ray & MRI Classification using Deep Learning

This project provides a **complete medical image classification pipeline** built with **FastAPI + PyTorch**, including:

- Automated dataset download
- Transfer learning (MobileNetV2, ResNet50)
- Training and evaluation with GradCAM visualization
- Performance metrics, confusion matrices, and ROC curves

---

## 🚀 Features

✅ Automated dataset setup  
✅ Transfer Learning support (MobileNetV2 / ResNet50 / Custom CNN)  
✅ Model evaluation with detailed metrics and visualizations  
✅ Configurable data directories and hyperparameters  
✅ Ready-to-integrate API for deployment

---

## 🧩 Project Structure

xray-mri-classification/
├── data/ # Dataset directory
├── models/ # Saved model checkpoints
├── results/ # Evaluation results and plots
├── scripts/ # Training, evaluation, and dataset utilities
│ ├── download_datasets.py
│ ├── setup_real_dataset.py
│ ├── explore_dataset.py
│ ├── train_model.py
│ ├── evaluate_model.py
├── src/ # Core package (configs, utils, services)
└── requirements.txt


---
```

## ⚙️ Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/amirpandit/xray-mri-classification.git
cd xray-mri-classification
````

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
# or
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Step 1: Download Dataset

To automatically download and organize the dataset:

```bash
python -m scripts.download_datasets
```

✅ This will create a `data/` directory with training, validation, and test sets.

---

## 🔍 Step 2: Explore Dataset (Optional)

You can visualize and analyze dataset statistics:

```bash
python -m scripts.explore_dataset --data-dir data --sample-size 200
```

This generates:

- Class distribution plots
- Image dimension analysis
- Augmentation recommendations

---

## 🧰 Step 3: Setup Real Dataset (Optional)

If you already have raw data in another folder:

```bash
python -m scripts.setup_real_dataset --source-dir path/to/raw_data --output-dir data
```

This script organizes your dataset into:

```
data/
 ├── train/
 ├── val/
 └── test/
```

---

## 🧠 Step 4: Train the Model

Run model training with transfer learning:

```bash
python -m scripts.train_model --data-dir data --model mobilenet_v2 --epochs 3 --batch-size 4
```

> 💡 For low-end hardware (like 8GB RAM + Intel i5), use small batch size and fewer epochs.

**Example safe command for limited hardware:**

```bash
python -m scripts.train_model --data-dir data --model mobilenet_v2 --epochs 1 --batch-size 4
```

Model checkpoints will be saved in the `models/` directory.

---

## 🧪 Step 5: Evaluate the Model

After training, evaluate your model’s performance:

```bash
python -m scripts.evaluate_model --model mobilenet_v2 --data-dir data --models-dir models --results-dir results
```

This will:

- Compute **Accuracy, Precision, Recall, F1-score**
- Generate **Confusion Matrix** & **ROC Curves**
- Save all results to the `results/` directory

---

## 📊 Results and Outputs

After running evaluation, check the `results/` folder:

```
results/
 ├── mobilenet_v2_confusion_matrix.png
 ├── mobilenet_v2_confusion_matrix_normalized.png
 ├── mobilenet_v2_roc_curve.png
 ├── mobilenet_v2_metrics.csv
 └── mobilenet_v2_evaluation_report.json
```

You can open the JSON or CSV files to inspect all evaluation metrics.

---

## 🧩 Supported Models

| Model          | Description                             |
| -------------- | --------------------------------------- |
| `mobilenet_v2` | Lightweight, ideal for limited hardware |
| `resnet50`     | Deep residual model for more accuracy   |
| `cnn`          | Custom CNN baseline model               |

---

## ⚡ Troubleshooting

| Problem                                              | Possible Fix                                           |
| ---------------------------------------------------- | ------------------------------------------------------ |
| ❌ `Training data not found at data/processed/train` | Run `scripts.setup_real_dataset` or check dataset path |
| ❌ CUDA out of memory                                | Reduce `--batch-size` or use CPU                       |
| ❌ Key mismatch in model loading                     | Remove `"model."` prefix or retrain model              |
| 💻 Slow training                                     | Use `--epochs 1 --batch-size 4` and lightweight model  |

---

## 🧠 Example Full Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Download dataset
python -m scripts.download_datasets

# 3. Train with MobileNetV2
python -m scripts.train_model --data-dir data --model mobilenet_v2 --epochs 1 --batch-size 4

# 4. Evaluate the model
python -m scripts.evaluate_model --model mobilenet_v2 --data-dir data --models-dir models --results-dir results
```

---

## 🏁 Done!

🎉 You’ve successfully trained and evaluated a medical image classifier using deep learning.
All results are stored in the `results/` folder for easy analysis and visualization.

---

**Developed by [Amir Pandit](https://www.amirpandit.com.np/#/about)**
_Built with PyTorch, FastAPI, and ❤️_

