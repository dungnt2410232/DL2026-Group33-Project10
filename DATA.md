# Dataset Information

## Sources and Versions

- **Questions and answers:** Official VQA v2.0, COCO val2014 validation partition.
- **Official download:** https://visualqa.org/download.html
- **Required files:**
  - `v2_OpenEnded_mscoco_val2014_questions.json`
  - `v2_mscoco_val2014_annotations.json`
- **Images:** `val2014-resised` from [henrychibueze/vqa-dataset](https://www.kaggle.com/datasets/henrychibueze/vqa-dataset).
- **Kaggle image dataset version:** Not recorded. This is separate from the notebook save version.
- Images are used as supplied; the source resize procedure was not independently verified.

The original `vaq2.0.*Images.txt` files are excluded because they contain only yes/no answers and their splits share images.

## Project Splits

Matching the official questions to available images produces a subset of **63,916 questions and 8,127 images**. This is not the full VQA v2 benchmark.

Sorted unique image IDs are shuffled with seed 42 and split approximately 80%/10%/10%. All questions for the same image remain together.

| Split | Questions | Images |
|---|---:|---:|
| Train | 51,107 | 6,501 |
| Dev | 6,684 | 812 |
| Test | 6,125 | 814 |

No image IDs overlap across splits. Duplicate image contents were not separately checked. These are project splits of the official validation partition, not official VQA train/test splits.

## Preprocessing

1. Join questions and annotations by `question_id` and verify matching `image_id`.
2. Match images using `COCO_val2014_{image_id:012d}.jpg`; retain only questions with available images and annotations.
3. Save the question, image path, representative answer, all annotator answers, and answer type.
4. Split by image ID and assign keyword-based categories: counting, color, spatial, reasoning, object, and `other`. Rules are implemented in `src/data_prep.py`; categories are not manually verified.
5. At loading time, convert images to RGB and use `ViltProcessor`, limiting questions to 40 tokens. Training answer matching uses lowercase and leading/trailing whitespace removal; evaluation also collapses internal whitespace.
6. Training excludes 688 examples with no supported annotator answer in the model's fixed vocabulary, leaving **50,419 training examples**. Dev and Test are evaluated without this filtering.

## Reproduction and Download

Add the image dataset and both official JSON files as Kaggle inputs. Place the repository's source scripts together in `/kaggle/working/src`, then run these commands in a notebook code cell:

```python
%cd /kaggle/working
!python src/data_prep.py --raw-dir /kaggle/input --output-dir /kaggle/working/vqa_processed --seed 42
```

This generates `train.jsonl`, `dev.jsonl`, and `test.jsonl` in `/kaggle/working/vqa_processed`. The files retain question IDs and image IDs identifying split membership. Use `src/dataset.py` for model input loading and `src/train.py` for training-only answer filtering.

**Processed data download:** Download `prepared_data.zip` from the [saved Kaggle Output version](https://www.kaggle.com/code/dungnt28/vilt-vqa/output?scriptVersionId=355785821).

The ZIP contains processed records, not images. Obtain images from the source above. Image paths may need updating if the input location changes. The saved Output must be accessible to reviewers.

## Checkpoint Exposure

The starting checkpoint, `dandelin/vilt-b32-finetuned-vqa`, is already fine-tuned on VQAv2. Image-disjoint project splits do not guarantee that Test examples were unseen during the checkpoint's earlier training.
