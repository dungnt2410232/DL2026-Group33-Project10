# Visual Question Answering Across Different Question Types

A deep learning project using **ViLT** to answer questions about images and evaluate performance across five question types:

- Object recognition
- Color
- Counting
- Spatial relationships
- Visual reasoning

## Model and Dataset

- **Model:** [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- **Questions and answers:** [Official VQA v2](https://visualqa.org/download.html), validation partition.
- **Images:** `val2014-resised` from the [Kaggle image source](https://www.kaggle.com/datasets/henrychibueze/vqa-dataset).
- Use only official questions with a matching available image; this is a subset, not the full VQA v2 dataset.
- **Environment:** Kaggle Notebook with a GPU
- Dataset details are provided in [DATA.md](DATA.md).

## Project Structure

```text
src/                 # Data preparation, training, evaluation, and inference
notebook/            # Kaggle notebooks
README.md
DATA.md
requirements.txt
```

## Setup on Kaggle

1. Create a Kaggle Notebook.
2. Enable a GPU accelerator and Internet for downloading the model.
3. Add the Kaggle image dataset and an input containing these official JSON files:
   - `v2_OpenEnded_mscoco_val2014_questions.json`
   - `v2_mscoco_val2014_annotations.json`
4. Upload `requirements.txt` as a notebook input and install it using its actual path:

```python
!pip install -r /kaggle/input/YOUR_INPUT_FOLDER/requirements.txt
```

Replace `YOUR_INPUT_FOLDER` with the uploaded input's folder name. Dataset files are available under `/kaggle/input/`; save generated files under `/kaggle/working/`.

## Experiment Workflow

1. Join official questions and answers, keeping records with available images.
2. Split this subset by image into 80% training, 10% validation, and 10% evaluation with seed 42.
3. Evaluate the initial ViLT checkpoint.
4. Fine-tune ViLT on the training split.
5. Evaluate overall and per-category accuracy.
6. Demonstrate predictions with an image and a question.

**Status:** Initial setup; code and results are not available yet. The final README will include the notebook link, run steps, settings, and measured results needed to reproduce the experiments.

**Note:** The initial checkpoint is already fine-tuned on VQAv2. Check dataset overlap before describing evaluation examples as unseen.
