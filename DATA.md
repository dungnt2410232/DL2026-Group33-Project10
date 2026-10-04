# Dataset Information

## Source

- **Dataset:** VQA Dataset
- **Uploader:** henrychibueze
- **Download:** https://www.kaggle.com/datasets/henrychibueze/vqa-dataset
- **Kaggle version and original dataset version:** To be confirmed after inspection.
- **Official original-source URL:** To be confirmed. If the dataset originates from VQA: https://visualqa.org/download.html

## Access

Add the dataset as an input to the Kaggle Notebook. Inspect the actual file names and paths under `/kaggle/input/` before loading them.

## Data Split

Preserve official splits if provided. Otherwise, split labeled data into **80% training, 10% validation, and 10% evaluation**, grouping questions by image and using seed **42**. Check duplicate images across splits. Final split sizes and IDs will be recorded after preparation.

## Preprocessing

1. Match images, questions, and answers using their IDs.
2. Check missing images, empty questions, and duplicate records.
3. Convert images to RGB and process images/questions with `ViltProcessor`.
4. Normalize answers consistently for training and evaluation.
5. Tag questions as object recognition, color, counting, spatial relationships, or visual reasoning; retain unmatched questions as `other`.

## Reproduction

Data preparation will be implemented in `src/data_prep.py` and run from the Kaggle Notebook. The script is not implemented yet. Record the dataset version, exact split IDs, category rules, and preprocessing settings with the experiment.

**Processed dataset download:** Pending. If a processed dataset is created, provide a downloadable link here, subject to the source's redistribution terms.
