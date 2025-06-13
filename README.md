# Fine-Grained Few-Shot Classification with Part Matching

Official code for the **FGVC 2025** workshop paper:  
**Fine-Grained Few-Shot Classification with Part Matching**  
📄 [Link to paper](https://your-paper-link.com)

We provide the official implementation of **Simple Matching Parts Learner (SMPL)**, our proposed model for fine-grained few-shot classification using part-level feature matching. Currently, we provide the data and code to replicate our experiments on the Hotels-8k dataset. Future data releases will be forthcoming.

---

## 📦 Downloads

### 🔹 Precomputed Part Features
Download precomputed DINO or DINOv2 part features for the Hotels-8K dataset here:

➡️ [Download Part Features](https://your-part-features-link.com)

Extract to:

After downloading, extract the file to the following path:
```
data/
├── features/
│   └── {dataset}/
│       └── features/
│           └── {dataset}_{part_encoder}.npy
```

This file is a serialized Python dictionary. Each key is an image path, and each value is another dictionary containing:
- `part_encodings`: the extracted part features
- `part_indices`: part labels that correspond to pixel values in the segmentation maps
### 🔹 Images and Segmentation Maps
To view the original images and the corresponding segmentation maps (generated with GroundedSAM):

➡️ [Download Images and Segmentations](https://your-image-and-maps-link.com)