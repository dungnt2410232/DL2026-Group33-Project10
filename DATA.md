# Dataset Information

## Sources

- **Questions and answers:** Official **VQA v2.0**, validation partition associated with COCO `val2014` images.
- **Official download:** https://visualqa.org/download.html
- **Files:**
  - `v2_OpenEnded_mscoco_val2014_questions.json`
  - `v2_mscoco_val2014_annotations.json`
- **Images:** `val2014-resised` from https://www.kaggle.com/datasets/henrychibueze/vqa-dataset
- **Kaggle image version and resize procedure:** Not yet verified; record before final submission.

The three original `vaq2.0.*Images.txt` files are excluded. Their answers are only yes/no, and their splits share images.

## Access

Add both the image dataset and the two official JSON files as inputs to the Kaggle Notebook. Files are read from `/kaggle/input/`.

## Project Splits

Use only official validation questions whose images exist in the Kaggle image folder. This creates a **subset of VQA v2 validation**, not the full benchmark.

Planned split: **80% training, 10% validation, 10% evaluation** by unique image, with seed **42**. Keep all questions from each image together and check duplicate image contents across splits. Final counts and split IDs will be saved after preparation. These are project splits, not official VQA train/test splits.

## Preprocessing

1. Join questions and annotations by `question_id`; check matching `image_id`.
2. Match each image ID to `COCO_val2014_{image_id:012d}.jpg` in `val2014-resised`.
3. Record missing images and invalid records; retain all human answers for valid examples.
4. Create image-grouped splits and assign the five project question categories; keep unmatched questions as `other`.
5. Convert images to RGB and use `ViltProcessor` for model inputs. Normalize answers consistently for training and evaluation.

## Reproduction

Implement preparation in `src/data_prep.py` and run it from the Kaggle Notebook. Save source versions/checksums, split IDs, category rules, and preprocessing settings. The script and processed data are not available yet.

**Processed dataset download:** Pending; add a versioned downloadable link after preparation, subject to redistribution terms.

**Checkpoint overlap:** ViLT's initial checkpoint is already fine-tuned on VQAv2. Creating project splits does not guarantee these examples were unseen by that checkpoint; document its prior data exposure when reporting results.
