# Visual Question Answering Across Different Question Types

A deep learning project using **ViLT** to answer image-based questions and compare performance across counting, color, spatial relationships, visual reasoning, and object recognition. Questions outside these keyword-based categories are labeled `other`.

## Model and Data

- **Starting checkpoint:** [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa), already fine-tuned on VQAv2, with 3,129 answer classes.
- **Questions and annotations:** [Official VQA v2](https://visualqa.org/download.html), COCO val2014 validation partition.
- **Images:** `val2014-resised` from the [Kaggle dataset](https://www.kaggle.com/datasets/henrychibueze/vqa-dataset).
- **Project subset:** 63,916 questions matched to 8,127 available images.
- **Image-disjoint split, seed 42:** Train 51,107 questions; Dev 6,684; Test 6,125. Training excludes 688 examples without supported answer labels, leaving 50,419 examples. Dev and Test remain complete.

See [DATA.md](DATA.md) for dataset versions, preprocessing, and processed-data access. The local split does not establish that Test examples were unseen by the starting checkpoint.

## Project Structure

```text
src/
├── data_prep.py       # Join data, split by image, assign categories
├── dataset.py         # Load images/questions and construct answer targets
├── model.py           # Load ViLT and its processor
├── train.py           # Fine-tune and select checkpoints using Dev
├── evaluate.py        # Overall and category-level evaluation
└── inference.py       # Predict an answer for an image and question
notebook/              # Kaggle notebooks, including additional experiments
README.md
DATA.md
requirements.txt
```

## Setup on Kaggle

1. Enable a GPU and Internet in the notebook settings. Experiments used a Tesla T4 and Transformers 4.57.6.
2. Add the image dataset and these official JSON files as inputs:
   - `v2_OpenEnded_mscoco_val2014_questions.json`
   - `v2_mscoco_val2014_annotations.json`
3. Upload the repository files as a Kaggle input. Copy `src/` and `requirements.txt` to `/kaggle/working/` using your input's actual path. Keep all source scripts together.
4. Run the following in notebook code cells:

```python
%cd /kaggle/working
!pip install -r requirements.txt
!pip install "transformers==4.57.6"
```

## Reproduce the Experiments

Run these commands in Kaggle code cells after setup. Generated data and outputs are saved under `/kaggle/working/`.

**Prepare data and evaluate the baseline:**

```python
!python src/data_prep.py --raw-dir /kaggle/input --output-dir /kaggle/working/vqa_processed --seed 42
!python src/evaluate.py --model dandelin/vilt-b32-finetuned-vqa --data-file /kaggle/working/vqa_processed/test.jsonl --device cuda --batch-size 8 --output-dir /kaggle/working/results/baseline
```

**Train Run A and evaluate its best Dev checkpoint:**

```python
!python src/train.py --model dandelin/vilt-b32-finetuned-vqa --data-dir /kaggle/working/vqa_processed --epochs 1 --batch-size 4 --accumulation-steps 4 --learning-rate 2e-5 --seed 42 --output-dir /kaggle/working/vilt_finetuned
!python src/evaluate.py --model /kaggle/working/vilt_finetuned/best --data-file /kaggle/working/vqa_processed/test.jsonl --device cuda --batch-size 8 --output-dir /kaggle/working/results/finetuned
```

**Additional runs:** use the same training command and change only the settings below. Each starts from the original checkpoint, not from Run A.

| Run | Epochs | Learning rate | Training output directory |
|---|---:|---:|---|
| B | 2 | 2e-5 | `/kaggle/working/vilt_run_B` |
| C | 1 | 1e-5 | `/kaggle/working/vilt_run_C` |

For evaluation, set `--model` to the corresponding output directory followed by `/best`, and save to `results/run_B` or `results/run_C`. The notebook includes the complete commands. The saved `best_checkpoint.json` identifies the selected epoch; a two-epoch run may select epoch 1.

## Test Results

All evaluations use 6,125 Test questions. Training uses batch size 4, four-step gradient accumulation, and seed 42. Run A is the main experiment; B and C are additional configuration comparisons.

| Model / Run | Exact Match (%) | Consensus (%) |
|---|---:|---:|
| Baseline | 72.88 | 82.50 |
| A: 1 epoch, 2e-5 | 73.22 | 82.74 |
| B: 2 epochs, 2e-5 | 72.64 | 82.53 |
| C: 1 epoch, 1e-5 | 73.67 | 83.13 |

Scores use lowercase and whitespace normalization, **not the full official VQA evaluation procedure**. C has the highest observed overall Test scores among these runs; configuration selection should use Dev results.

Evaluation saves `metrics.json`, `metrics.csv`, `predictions.jsonl`, and `vqa_predictions.json`. Category labels are heuristic and group sizes differ.

## Demo and Saved Outputs

The notebook demonstrates Run A predictions with images, questions, and reference answers. For a custom image, replace `photo.jpg` with its actual path:

```python
!python src/inference.py --model /kaggle/working/vilt_finetuned/best --image photo.jpg --question "What color is the car?"
```

- [Kaggle notebook](https://www.kaggle.com/code/dungnt28/vilt-vqa)
- **Saved outputs / downloads:** (https://www.kaggle.com/code/dungnt28/vilt-vqa/output?scriptVersionId=355785821) — add the saved Kaggle Output version link containing the submitted checkpoints, processed data, and results.

