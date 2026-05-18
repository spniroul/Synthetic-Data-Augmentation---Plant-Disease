# 🌿 Synthetic Data Augmentation for Imbalanced Plant Disease Classification

> Using Denoising Diffusion Probabilistic Models (DDPMs) to generate synthetic minority-class images and improve plant disease classification under severe class imbalance.

**CS 444 Final Project — University of Illinois Urbana-Champaign, May 2026**  
*Bhaksar Aryal · Soujan Niroula*

---

## 📋 Overview

Real-world agricultural image datasets are naturally imbalanced. Rare disease classes are underrepresented, causing classifiers to favor majority classes and perform poorly on the minority disease types that are most critical to detect.

This project investigates whether **DDPMs** can generate high-quality synthetic images for minority disease classes to improve **EfficientNet-B0** classifier performance under controlled class imbalance (10:1 and 50:1 ratios), compared to classical remedies including oversampling, data augmentation, and weighted loss functions.

---

## 🔬 Key Research Questions

- How severely does class imbalance (10:1 and 50:1) degrade EfficientNet-B0 performance?
- How well do classical remedies recover minority class recall?
- Can DDPM-generated images outperform classical methods, especially at extreme imbalance?

---

## 📊 Dataset

**PlantVillage** ([HuggingFace](https://huggingface.co/datasets/mohanty/PlantVillage)) — 6 tomato disease classes, 39,096 total images.

| Class | Images |
|---|---|
| Yellow Leaf Curl Virus *(majority)* | 16,071 |
| Late Blight | 5,727 |
| Septoria Leaf Spot *(minority)* | 5,313 |
| Healthy | 4,773 |
| Target Spot | 4,212 |
| Early Blight *(minority)* | 3,000 |

A stratified **70/15/15** train/val/test split is applied. Only the training set is modified for imbalance experiments.

**Controlled imbalance conditions:**

| Condition | Majority samples | Minority samples |
|---|---|---|
| 10:1 | 983 | 98 per minority class |
| 50:1 | 983 | 19 per minority class |

---

## 🏗️ Methods

### Classifier — EfficientNet-B0
- Pretrained on ImageNet, fine-tuned end-to-end
- Input: 224×224, ImageNet normalization
- Optimizer: Adam with cosine LR schedule
- Checkpoint selected by best validation Macro F1

### Classical Baselines
1. **No Augmentation** — baseline imbalanced training
2. **Classical Augmentation** — random geometric/color transforms
3. **Oversampling** — random duplication of minority samples
4. **Weighted Loss** — inverse-frequency class weights in cross-entropy

### DDPM-Based Synthetic Augmentation
A **U-Net DDPM** is trained on minority class images (T=1000 timesteps). Architecture features residual blocks, self-attention at 8×8 resolution, group normalization, and sinusoidal timestep embeddings.

Two conditions are evaluated:
- **Real+Synthetic** — real minority samples + DDPM-generated images, combined with majority real data
- **Synthetic Only** — classifier trained solely on DDPM-generated images (ablation)

After training, minority class counts increase from **98 → 598** (10:1) and **19 → 519** (50:1).

---

## 📈 Results

### Balanced Benchmark (Upper Bound)
| Accuracy | Macro F1 | Minority Recall | Balanced Acc. |
|---|---|---|---|
| 0.9867 | 0.9798 | 0.9067 | 0.9772 |

### Full Comparison

| Method | Imbalance | Accuracy | Macro F1 | Min. Recall | Bal. Acc. |
|---|---|---|---|---|---|
| **Balanced Benchmark** | — | 0.9867 | **0.9798** | **0.9067** | **0.9772** |
| No Augmentation | 10:1 | 0.9494 | 0.9213 | 0.7200 | 0.9189 |
| No Augmentation | 50:1 | 0.8665 | 0.7842 | 0.3333 | 0.8004 |
| Classical Aug. | 10:1 | 0.8542 | 0.7195 | 0.0467 | 0.7530 |
| Classical Aug. | 50:1 | 0.7918 | 0.6015 | 0.0451 | 0.6782 |
| Oversampling | 10:1 | 0.9391 | 0.9019 | 0.6467 | 0.9008 |
| Oversampling | 50:1 | 0.8542 | 0.7635 | 0.2600 | 0.7708 |
| **Weighted Loss** | **10:1** | 0.9545 | **0.9307** | **0.7200** | **0.9244** |
| **Weighted Loss** | **50:1** | 0.8276 | 0.7694 | **0.4662** | 0.7904 |
| **DDPM Aug (Real+Syn)** | **10:1** | 0.9427 | **0.9120** | 0.7000 | 0.9093 |
| **DDPM Aug (Real+Syn)** | **50:1** | **0.8916** | **0.8222** | 0.4267 | **0.8264** |
| Synthetic Only | 10:1 | 0.7908 | 0.5903 | 0.0000 | 0.6733 |
| Synthetic Only | 50:1 | 0.7969 | 0.6155 | 0.0267 | 0.6812 |

> Bold entries = best result per metric per imbalance level (excluding balanced benchmark).

---

## 🔍 Key Findings

1. **Imbalance severely degrades minority recall** while preserving misleadingly high accuracy — minority recall drops from 0.907 (balanced) to 0.333 (50:1, no augmentation). Macro F1 and minority recall are necessary metrics.

2. **Weighted loss is the strongest single method** — best minority recall at both imbalance levels while maintaining high Macro F1.

3. **DDPM Aug (Real+Synthetic) is strongest at 50:1 on Macro F1 and balanced accuracy** — outperforms all classical methods except weighted loss on minority recall, and beats weighted loss on accuracy (0.892 vs. 0.828) and balanced accuracy (0.826 vs. 0.790).

4. **Classical augmentation unexpectedly fails** — with only 19–98 real minority images, augmentations exhaust quickly, producing overfitted variations that score *worse* than no augmentation on minority recall.

5. **Synthetic-only training is ineffective** — DDPM models trained on very few low-resolution images cannot substitute for real training data (near-zero minority recall).

6. **Most promising direction: combining DDPM augmentation with weighted loss** — not evaluated here, left for future work.

---

## ⚠️ Limitations

- DDPM trained at **64×64 resolution** — higher resolution (128×128 or 256×256) would likely improve synthetic image quality and downstream performance
- No **FID / Inception Score** computed for synthetic image quality assessment
- Fixed real-to-synthetic ratio — sweeping this may improve results
- Single backbone (EfficientNet-B0) — results may differ with deeper architectures
- Weighted loss + DDPM combination not evaluated

---

## 🛠️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/plant-disease-ddpm.git
cd plant-disease-ddpm

# Install dependencies
pip install -r requirements.txt

# Download PlantVillage dataset
python scripts/download_data.py

# Train DDPM on minority classes
python train_ddpm.py --epochs 200 --resolution 64

# Generate synthetic images
python generate.py --num_samples 500 --output_dir synthetic/

# Train EfficientNet-B0 classifier
python train_classifier.py --method ddpm_aug --imbalance_ratio 10

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt
```

---

## 📁 Project Structure

```
├── data/                   # Dataset loading and imbalance creation
├── models/
│   ├── ddpm/               # U-Net DDPM implementation
│   └── classifier/         # EfficientNet-B0 fine-tuning
├── scripts/
│   ├── download_data.py
│   └── create_imbalance.py
├── train_ddpm.py           # DDPM training loop
├── generate.py             # Synthetic image generation
├── train_classifier.py     # Classifier training (all methods)
├── evaluate.py             # Test set evaluation
├── requirements.txt
└── README.md
```

---

## 📚 References

- Mohanty et al., "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, 2016.
- Ho et al., "Denoising diffusion probabilistic models," *NeurIPS*, 2020. [[arXiv]](https://arxiv.org/abs/2006.11239)
- Tan & Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," *ICML*, 2019.

---

## 👥 Contributors

| Contributor | Responsibilities |
|---|---|
| Bhaksar Aryal | Visualization, proposal writing, analysis, report writing |
| Soujan Niroula | DDPM implementation, synthetic image generation, data pipeline, EfficientNet-B0 training, classical baselines|

---

## 📄 License

This project is for academic use only. See [LICENSE](LICENSE) for details.
