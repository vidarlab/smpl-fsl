# Fine-Grained Few-Shot Classification with Part Matching

Official code for the **FGVC 2025** workshop paper:  
📄 [Fine-Grained Few-Shot Classification with Part Matching](https://openaccess.thecvf.com/content/CVPR2025W/FGVC/papers/Black_Fine-grained_Few-Shot_Classification_with_Part_Matching_CVPRW_2025_paper.pdf)

<img width="1224" alt="image" src="https://github.com/user-attachments/assets/5caba300-cc74-40e3-860e-75afe271bd19" />


This repository contains the official implementation of **Simple Matching Parts Learner (SMPL)**, our proposed method for fine-grained few-shot classification using part-level feature matching. We currently provide code to replicate our experiments on the Hotels-8K dataset, with additional datasets to be released in the future.

---

## 📦 Downloads

### 🔹 Precomputed Part Features
We provide precomputed part features for the Hotels-8K dataset using either DINO or DINOv2 as the Part Encoder:

➡️ [Dino](https://tuprd-my.sharepoint.com/:u:/g/personal/tul03156_temple_edu/EW4w48jCEqNMtcs4WW_7fToBCOKKkkEh1IHBZDzlU1i7Pg?e=7jxeSo)

➡️ [DinoV2](https://tuprd-my.sharepoint.com/:u:/g/personal/tul03156_temple_edu/ERm3y4p0rjNFnCRvCNoj_3YByQO99wW_D35BFVqvRN9l4g?e=ATEac8)

Extract to:

After downloading, extract the file to the following path:
```
data/
│   └── {dataset}/
│       └── features/
│           └── {dataset}_{part_encoder}.npy
```

- `{dataset}`: e.g., `hotels`
- `{part_encoder}`: either `dino` or `dinov2`

This .npy file is a serialized Python dictionary containing part features for the training, validation, and test sets. Each entry maps an image path to a nested dictionary with two keys:
- `part_features` → a NumPy array of shape `(N + 1, D)`, where `N` is the number of distinct parts in the image and `D` is the feature dimensionality. The last row is the global image feature.
- `part_indices` → a NumPy array of shape `(N + 1,)`, where each value corresponds to the part ID in the segmentation mask. The final value (for the global feature) is always `1000`.

### 🖼️ Original Images & Segmentations

You can download the original images and segmentation maps (generated using GroundedSAM) here:

➡️ [Download Images and Segmentations](https://tuprd-my.sharepoint.com/:u:/g/personal/tul03156_temple_edu/EbqB3f6-AWNItgBWYy9akM0BqlHGn-g36Pim6G6zf6bE5w?e=U2Mz8S)

<div align="center">
  <img width="240" src="https://github.com/user-attachments/assets/5bbb409c-4e07-4dba-9fab-23993d97fa84" />
  <img width="240" src="https://github.com/user-attachments/assets/b99e8778-b052-472b-9dee-5b32f9c051c7" />
</div>

---

## 🚀 Training The Part Matcher

To train the Part Matcher model, run:

```
python train.py –data_dir data –dataset {dataset} –part_encoder {dino|dinov2}
```

Training episodes are generated stochastically, sampling between 5–20 ways and 1–5 shots per episode, as described in the paper. Class splits used for episode generation are located at:

```
data/{dataset}/splits/
```

By default, the best checkpoint will be saved to:

```
checkpoints/{dataset}/{part_encoder}/best.pt
```

## 📊 Evaluation

To evaluate on 20-way classificatio tasks, run:

```
python evaluation.py 
–dataset {dataset} 
–part_encoder {dino|dinov2} 
–checkpoint {path_to_model} 
–n_shots {1|5}
```

Evaluation episodes are pre-generated and stored at:
```
data/{dataset}/test_episodes/20way_{n_shots}shots.json
```
## 📖 Citation

If you use our work, please cite:

```bibtex
@InProceedings{Black_2025_CVPR,
    author    = {Black, Samuel and Souvenir, Richard},
    title     = {Fine-grained Few-Shot Classification with Part Matching},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {2057-2067}
}

