# Benchmarking Deep Convolutional Neural Networks, Grad-CAM Interpretability, and Adversarial FGSM Robustness for Pediatric Pneumonia Diagnosis

**Author:** Rijan Rayamajhi  
**Affiliation:** Medical Image Computing & Computer-Assisted Diagnosis Research Group  
**Repository:** [github.com/rijan-rayamajhi/chest-xray-pneumonia-classifier](https://github.com/rijan-rayamajhi/chest-xray-pneumonia-classifier)  
**Date:** August 2026  

---

## Abstract

Automated classification of chest radiographs plays a crucial role in triaging pulmonary infections in pediatric populations. However, existing CAD algorithms often suffer from validation set sampling bias, unquantified statistical uncertainty, black-box decision making, and vulnerability to adversarial noise. In this study, we present a systematic benchmark of five deep convolutional neural network architectures—**BaselineCNN**, **DenseNet121**, **ResNet50V2**, **EfficientNetB0**, and **MobileNetV2**—evaluated on a dataset of 5,856 pediatric chest radiographs. To prevent validation split bias present in raw benchmark partitions, we implement a stratified 80/20 train-validation strategy alongside inverse class loss weighting. Model performance is rigorously quantified on an independent test set ($N=624$) using non-parametric empirical bootstrapping ($N=500$) to derive **95% Confidence Intervals (CI)** across Accuracy, Sensitivity (Recall), Specificity, Precision, ROC-AUC, and F1-Weighted metrics. **DenseNet121** achieved the highest overall diagnostic performance with an **ROC-AUC of 0.9633 (95% CI: 0.9483 – 0.9758)** and an **Accuracy of 0.9054 (95% CI: 0.8814 – 0.9295)**. To evaluate trustworthy AI security, we tested the best model against **Fast Gradient Sign Method (FGSM)** adversarial attacks across perturbation scales $\epsilon \in [0.0, 0.05]$. We observe that minor bounded perturbations ($\epsilon = 0.005$) degrade overall classification accuracy down to $6.80\%$, shifting Grad-CAM attention heatmaps away from parenchymal lung opacities toward non-pathological background areas. These findings demonstrate the necessity of adversarial robustness verification prior to clinical deployment.

*Keywords:* Chest Radiography, Pediatric Pneumonia, Deep Transfer Learning, Grad-CAM Explainability, Adversarial Robustness, FGSM Attack, Bootstrapped Confidence Intervals, Trustworthy AI.

---

## 1. Introduction

Pneumonia remains the single largest infectious cause of death in children under five years worldwide, accounting for over 700,000 infant fatalities annually (WHO, 2022). Prompt and accurate diagnosis is critical for initiating antimicrobial therapy. While anterior-posterior (AP) chest radiography is the gold-standard imaging modality for diagnosing pulmonary consolidation, diagnostic interpretation requires specialized pediatric radiological expertise, which is severely limited in low-resource and emergency clinical settings.

Computer-Assisted Diagnosis (CAD) leveraging deep convolutional neural networks (CNNs) has emerged as a powerful tool to assist clinicians. However, deployment of CAD algorithms in clinical workflows faces key trust and security barriers:

1. **Validation Split Sampling Bias**: Published benchmarks frequently utilize raw unstratified dataset partitions containing inadequate validation sample sizes (e.g., 16 images), leading to severe evaluation instability.
2. **Unquantified Statistical Uncertainty**: Standard reporting often presents single-point estimates on test sets without reporting confidence intervals, obscuring variance and statistical significance across models.
3. **Black-Box Decision-Making**: Deep networks often risk learning spurious background correlations (e.g., hospital markers, clavicle alignment) rather than pathological lung features.
4. **Adversarial & Sensor Noise Vulnerability**: Deep clinical models are susceptible to imperceptible image noise or adversarial perturbations, which can drastically shift diagnostic output without triggering visual detection by human radiologists.

To resolve these challenges, this paper makes the following contributions:
- Implements a re-balanced, stratified 80/20 train-validation split across 5,856 pediatric chest radiographs with inverse class loss weighting.
- Benchmarks five distinct deep learning backbones ranging from custom shallow CNNs to deep residual and dense feature-reuse networks.
- Reports comprehensive diagnostic performance metrics accompanied by 95% empirical bootstrapped confidence intervals ($N=500$).
- Applies Grad-CAM visualization to qualitative true positive, true negative, false positive, and false negative diagnostic cases.
- **Formulates an FGSM adversarial robustness pipeline** ($\epsilon \in [0.0, 0.05]$) to evaluate accuracy degradation and quantify Grad-CAM attention heatmap shifts under gradient-based perturbation attacks.

---

## 2. Related Work

The open-access pediatric chest X-ray dataset published by Kermany et al. (2018) has served as a benchmark for automated pneumonia detection. Several research efforts have explored deep learning applications on this corpus:

- **Kermany et al. (2018)** established the baseline transfer learning framework on pediatric radiographs using InceptionV3, achieving an accuracy of 92.8% on binary classification.
- **Rajpurkar et al. (2017)** introduced *CheXNet*, a 121-layer DenseNet trained on over 100,000 chest radiographs, demonstrating that dense feature reuse architectures outperform radiologists in detecting pneumonia.
- **Stephen et al. (2019)** evaluated custom convolutional neural networks trained from scratch, highlighting the susceptibility of shallow architectures to overfit on smaller sample sizes without transfer pretraining.
- **Saraiva et al. (2019)** compared deep learning classifiers on chest X-rays, underscoring the necessity of class-imbalance weighting during gradient descent.
- **Selvaraju et al. (2017)** formulated Gradient-weighted Class Activation Mapping (Grad-CAM), enabling visual localization of target predictions without requiring structural architectural modifications or re-training.
- **Goodfellow et al. (2014) & Finlayson et al. (2019)** demonstrated that deep learning medical classifiers are extraordinarily fragile to Fast Gradient Sign Method (FGSM) adversarial attacks, where pixel-level noise invisible to the human eye completely subverts diagnostic classifications.

---

## 3. Methodology

### 3.1 Dataset Description and Re-balancing Strategy
The dataset comprises $5,856$ pediatric chest radiographs categorized into two anatomical classes: `NORMAL` ($1,583$ images) and `PNEUMONIA` ($4,273$ images). 

We merged the raw training and validation sets ($5,232$ total images) and performed a stratified 80/20 split (`SEED=42`), resulting in:
- **Training Set**: $4,185$ images ($1,073$ Normal, $3,112$ Pneumonia)
- **Validation Set**: $1,047$ images ($269$ Normal, $778$ Pneumonia)
- **Test Set**: $624$ images ($234$ Normal, $390$ Pneumonia)

To address class imbalance, inverse class weights were applied during cross-entropy loss optimization:
$$w_0 = \frac{N}{2 \cdot N_0} \approx 1.939, \quad w_1 = \frac{N}{2 \cdot N_1} \approx 0.674$$

### 3.2 Model Architectures
We evaluated five model backbones:
1. **BaselineCNN**: Custom 4-block CNN architecture with Batch Normalization, Max Pooling, and Dropout.
2. **DenseNet121**: Pretrained ImageNet backbone featuring dense feature reuse.
3. **ResNet50V2**: Residual network utilizing identity shortcut connections.
4. **EfficientNetB0**: Network scaled systematically using compound scaling coefficients.
5. **MobileNetV2**: Light-weight architecture utilizing inverted residual blocks.

### 3.3 Fast Gradient Sign Method (FGSM) Adversarial Formulation
To evaluate trustworthy AI security under input perturbations, we formulate an adversarial attack generator using the Fast Gradient Sign Method (FGSM):
$$X_{\text{adv}} = \text{clip}\left( X + \epsilon \cdot \text{sign}\left( \nabla_X J(\theta, X, y) \right), 0, 1 \right)$$
where $X$ is the input image tensor normalized to $[0, 1]$, $y$ is the ground-truth binary label, $\theta$ represents model weights, $J$ is the binary cross-entropy loss function, and $\epsilon$ controls the perturbation magnitude ($\epsilon \in \{0.0, 0.005, 0.01, 0.02, 0.05\}$).

---

## 4. Experimental Results

### 4.1 Quantitative Performance Comparison
The diagnostic metrics evaluated on the test set ($N=624$) with 95% bootstrapped confidence intervals are detailed in Table 1.

**Table 1: Benchmark diagnostic performance across 5 model architectures on independent test set ($N=624$) with 95% Empirical Bootstrapped Confidence Intervals ($N=500$).**

| Model | Accuracy (95% CI) | Sensitivity / Recall (95% CI) | Specificity (95% CI) | Precision (95% CI) | ROC-AUC (95% CI) | F1-Weighted (95% CI) |
|---|---|---|---|---|---|---|
| **DenseNet121** ⭐ | **0.9054** (0.8814 – 0.9295) | 0.9436 (0.9199 – 0.9638) | **0.8419** (0.7925 – 0.8912) | **0.9086** (0.8777 – 0.9371) | **0.9633** (0.9483 – 0.9758) | **0.9048** (0.8797 – 0.9292) |
| **ResNet50V2** | 0.8830 (0.8590 – 0.9054) | 0.9564 (0.9376 – 0.9740) | 0.7607 (0.7080 – 0.8191) | 0.8695 (0.8393 – 0.9010) | 0.9475 (0.9273 – 0.9652) | 0.8805 (0.8545 – 0.9041) |
| **MobileNetV2** | 0.8429 (0.8141 – 0.8718) | **0.9897** (0.9796 – 0.9975) | 0.5983 (0.5380 – 0.6580) | 0.8042 (0.7700 – 0.8398) | 0.9636 (0.9481 – 0.9767) | 0.8324 (0.8010 – 0.8636) |
| **EfficientNetB0** | 0.8446 (0.8157 – 0.8718) | 0.9487 (0.9254 – 0.9688) | 0.6709 (0.6099 – 0.7286) | 0.8277 (0.7931 – 0.8613) | 0.9405 (0.9194 – 0.9583) | 0.8391 (0.8077 – 0.8689) |
| **BaselineCNN** | 0.7628 (0.7324 – 0.7949) | 0.9718 (0.9555 – 0.9869) | 0.4145 (0.3610 – 0.4788) | 0.7345 (0.6956 – 0.7731) | 0.9088 (0.8878 – 0.9307) | 0.7356 (0.6991 – 0.7721) |

### 4.2 Adversarial FGSM Robustness Degradation
Figure 1 details the impact of increasing FGSM perturbation magnitude $\epsilon$ on model diagnostic performance.

![FGSM Robustness Degradation](outputs/figures/fgsm_robustness_degradation.png)  
*Figure 1: Accuracy degradation curve of DenseNet121 under FGSM adversarial perturbations ($\epsilon \in [0.0, 0.05]$).*

- At $\epsilon = 0.000$, DenseNet121 achieves baseline accuracy of $85.20\%$ on the evaluation subset.
- At $\epsilon = 0.005$, accuracy plummets to **$6.80\%$**, demonstrating extreme sensitivity to gradient-based noise.

### 4.3 Grad-CAM Attention Shift Under Attack
Figure 2 visualizes the structural shift in Grad-CAM activation heatmaps under adversarial attack.

![Grad-CAM Heatmap Shift Under FGSM](outputs/figures/fgsm_gradcam_shift.png)  
*Figure 2: Grad-CAM attention map distortion under increasing FGSM perturbation levels ($\epsilon = 0.00, 0.01, 0.03$).*

Under adversarial noise, the model's spatial attention shifts away from anatomical lung fields to non-pathological peripheral borders, revealing how unfortified CNNs can be subverted by minor perturbations.

---

## 5. Discussion and Limitations

### 5.1 Implications for Trustworthy Medical AI
Our adversarial evaluation highlights a critical vulnerability in clinical deep learning. While models like DenseNet121 achieve high standard accuracy ($90.54\%$), their vulnerability to FGSM attacks underscores the necessity of implementing adversarial training and robust loss functions before clinical deployment.

---

## 6. Conclusion

We presented a comprehensive medical image computing benchmark combining multi-model transfer learning, 95% bootstrapped confidence intervals, Grad-CAM explainability, and FGSM adversarial robustness testing. These results provide an empirical baseline for trustworthy CAD development.

---

## References

1. **Kermany, D. S., et al.** (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.
2. **Rajpurkar, P., et al.** (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv preprint arXiv:1711.05225*.
3. **Selvaraju, R. R., et al.** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *IEEE International Conference on Computer Vision (ICCV)*, 618–626.
4. **Finlayson, S. G., et al.** (2019). Adversarial attacks on medical machine learning. *Science*, 363(6433), 1287–1289.
5. **Goodfellow, I. J., Shlens, J., & Szegedy, C.** (2014). Explaining and Harnessing Adversarial Examples. *arXiv preprint arXiv:1412.6572*.
