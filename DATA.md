# VQA-Project_
# Dataset Documentation: VQA Dataset

## 1. Official Dataset Source & Version
- **Dataset Source:** [Kaggle VQA Dataset](https://www.kaggle.com/datasets/henrychibueze/vqa-dataset)
- **Base Benchmark:** Visual Question Answering (VQA) v2.0
- **Version:** Subset v2.0

## 2. Data Split
- **Train split:** 80% used for model fine-tuning.
- **Validation / Test split:** 20% reserved for detailed evaluation across question categories.

## 3. Preprocessing Procedure & Question Tagging
Questions are categorized into 5 types based on keywords:
- **Color:** Questions containing keywords like `what color`, `color of`.
- **Counting:** Questions starting with `how many`, `number of`.
- **Spatial Relationships:** Positional questions (`where is`, `next to`, `above`, `under`, `left`, `right`).
- **Object Recognition:** Entity-related questions (`what is`, `is there`, `does the`).
- **Visual Reasoning:** Contextual reasoning questions (`why`, `what is the person doing`).

## 4. Scripts to Reproduce Data
```bash
kaggle datasets download -d henrychibueze/vqa-dataset
unzip vqa-dataset.zip -d data/
python src/data_prep.py