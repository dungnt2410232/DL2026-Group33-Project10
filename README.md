# Visual Question Answering Across Different Question Types

A Deep Learning project evaluating the performance of the **BLIP** (`Salesforce/blip-vqa-base`) model across **5 distinct question types**:

- **Object Recognition**
- **Color**
- **Counting**
- **Spatial Relationships**
- **Visual Reasoning**

---

## 📁 Project Structure

```text
├── data/                  # Dataset directory
├── src/                   # Source code
│   ├── data_prep.py       # Download and tag question categories
│   ├── dataset.py         # Custom PyTorch Dataset
│   ├── model.py           # Model initialization
│   ├── train.py           # Fine-tuning loop
│   ├── evaluate.py        # Per-category accuracy evaluation
│   └── inference.py       # Quick demo script
├── DATA.md                # Dataset details & preprocessing rules
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

---

## ⚙️ Environment Setup

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Preprocess Data

Run the data preprocessing script:

```bash
python src/data_prep.py
```

This step downloads the dataset and assigns each question to one of the five question categories.

---

### Step 2: Fine-tune the Model

Fine-tune the **BLIP VQA model** using:

```bash
python src/train.py --epochs 3 --batch_size 16
```

You can modify the number of epochs and batch size according to your hardware.

---

### Step 3: Evaluate the Model

Evaluate the model's performance across the five question types:

```bash
python src/evaluate.py
```

The evaluation script reports accuracy for each question category, allowing you to compare the model's performance on different types of visual questions.

---

### Step 4: Demo Inference

Run a quick inference example with an image and a question:

```bash
python src/inference.py \
    --image "sample.jpg" \
    --question "What color is the car?"
```

The model will process the image and question and return its predicted answer.

---

## 📊 Question Types

| Question Type | Description | Example |
|---|---|---|
| **Object Recognition** | Identifies objects in an image | `What is the animal?` |
| **Color** | Determines the color of an object | `What color is the car?` |
| **Counting** | Counts objects in an image | `How many dogs are there?` |
| **Spatial Relationships** | Understands the position or relationship between objects | `What is next to the car?` |
| **Visual Reasoning** | Requires reasoning based on visual information | `What is the person likely doing?` |

---

## 🤖 Model

This project uses:

```text
Salesforce/blip-vqa-base
```

**BLIP (Bootstrapping Language-Image Pre-training)** is a vision-language model capable of answering questions based on visual information contained in an image.

The model is fine-tuned on the project dataset to improve its performance across different question types.

---

## 📈 Evaluation

The model is evaluated separately for each question category:

```text
Object Recognition      → Accuracy
Color                   → Accuracy
Counting                → Accuracy
Spatial Relationships   → Accuracy
Visual Reasoning        → Accuracy
```

This makes it possible to identify which types of visual questions the model handles well and which types remain challenging.

---

## 📚 Dataset

For detailed information about the dataset, preprocessing rules, and question-category definitions, see:

```text
DATA.md
```

---

## 🛠️ Technologies

- Python
- PyTorch
- Hugging Face Transformers
- BLIP
- Deep Learning
- Visual Question Answering (VQA)