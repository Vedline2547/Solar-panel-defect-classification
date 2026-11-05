# ☀️ Solar Panel Defect Classification Using Deep Learning

An end-to-end computer vision project that uses **Deep Learning and Transfer Learning** to identify and classify defects in solar panel images. The application uses an **EfficientNetB0** model to classify solar panels into six categories and provides predictions through an interactive Streamlit web application.

The application is deployed on **AWS EC2**, making the trained model accessible through a web browser.

## 🚀 Project Overview

Solar panel defects can reduce energy efficiency, affect performance, and increase maintenance costs. This project explores how deep learning can automate the identification of common solar panel conditions and defects using image classification.

The model analyzes an uploaded solar panel image and returns:
- Predicted defect or condition.
- Prediction confidence score.
- Top 3 predicted classes.
- Probability distribution across all six classes.

## 🎯 Classification Categories

The model identifies six classes:

| Class | Description |
|---|---|
| Bird-drop | Solar panels affected by bird droppings |
| Clean | Solar panels with no visible defects |
| Dusty | Solar panels covered with dust |
| Electrical-Damage | Solar panels showing electrical damage |
| Physical-damage | Solar panels with physical defects |
| Snow-Covered | Solar panels covered with snow |

## 🧠 Model Architecture

This project uses **EfficientNetB0**, a convolutional neural network architecture known for balancing model performance and computational efficiency.

### Techniques Used
- Transfer Learning
- Fine-tuning
- Image preprocessing
- Multi-class image classification
- Model evaluation

The trained model is saved as:

`trained_effnet_finetune.h5`

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Model development and fine-tuning |
| EfficientNetB0 | Image classification architecture |
| NumPy | Numerical operations |
| Pillow (PIL) | Image loading and processing |
| Streamlit | Interactive web application |
| AWS EC2 | Cloud deployment |
| Linux / Ubuntu | Server environment |
| Git & GitHub | Version control |

## 📂 Project Structure

```text
Solar-panel-defect-classification/
│
├── app.py
├── trained_effnet_finetune.h5
├── requirements.txt
├── README.md
└── .gitignore
```

*Additional training scripts, notebooks, or dataset directories can be included depending on your final repository structure.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Solar-panel-defect-classification.git
```

### 2. Navigate to the project directory

```bash
cd Solar-panel-defect-classification
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

## ☁️ AWS EC2 Deployment

The Streamlit application was deployed on an **AWS EC2 Ubuntu instance**.

Deployment involved:

1. Launching and configuring an EC2 instance.
2. Connecting to the instance using PuTTY (SSH).
3. Transferring project files
