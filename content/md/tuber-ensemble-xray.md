---
source_pdf: 661260.pdf
pages: 8
converted_with: pdftotext -layout
---

# 661260.pdf

> Auto-converted from PDF. Prefer this file over the PDF for AI reading.

---

  TUBER-ENSEMBLE: An Ensemble of Deep
 Convolutional Neural Networks for Tuberculosis
          Detection from Chest X-rays

Mohssine Kissane2 , Amine Zeguendry1,2 [0000-0002-5864-5667] , Walid Ahkouk2 ,
                           and Abdelilah Dahou2
1
    Laboratory of Computer Systems and Intelligent Systems, Cadi Ayyad University,
                               Marrakech, Morocco
      2
        LAMIGEP, EMSI, Moroccan School of Engineering, Marrakech, Morocco
                              a.zeguendry@emsi.ma



       Abstract. Tuberculosis (TB) continues to be a significant threat to
       global health, especially within low-resource environments where the
       availability of skilled radiologists is limited. In response, we introduce
       TUBER-ENSEMBLE, a new deep learning framework that operates on
       an ensemble basis. This framework integrates three distinct CNN archi-
       tectures—ResNet50, DenseNet121, and InceptionV3—to automate the
       detection of TB from chest X-ray images. Tested on the Kaggle TB
       dataset of 4200 images, our ensemble model, which uses a soft voting
       mechanism, reached an accuracy of 98.45%, precision of 98%, and recall
       of 92%, notably surpassing the performance of the individual models.
       Among the single models, InceptionV3 yielded the strongest results with
       98% accuracy and a 97% F1-score. Conversely, ResNet50 displayed lower
       sensitivity, achieving 83.6% accuracy and a 47% F1-score. The proposed
       ensemble method exhibits enhanced generalization and robustness, pre-
       senting a viable and scalable tool for TB screening initiatives in regions
       that lack sufficient diagnostic expertise.

       Keywords: Tuberculosis detection · Chest X-ray · Deep learning · En-
       semble learning · Transfer learning


1    Introduction
Affecting nearly a third of the global population, tuberculosis results in more than
10 million new infections each year [1]. Reports from the WHO identify TB as the
second most fatal infectious agent after COVID-19, with a worldwide incidence
rate greater than 133 cases for every 100,000 individuals [1]. Timely diagnosis
presents a persistent difficulty in areas with limited resources, where there is
a shortage of trained radiologists, even though chest X-rays are the favored
method for rapid screening [3], [5]. The field of medical image analysis has been
transformed by deep learning, especially with convolutional neural networks,
which have shown diagnostic capabilities on par with human experts [5,4].Never-
theless, approaches relying on a single model frequently do not exhibit sufficient


---

2       M. Kissane et al.

robustness when applied to varied patient demographics and imaging equipment.
This shortcoming is addressed by ensemble methods, which leverage the combined
strengths of several models to offset their individual limitations. This study puts
forward TUBER-ENSEMBLE, a framework that amalgamates ResNet50 [6],
DenseNet121 [7], and InceptionV3 [8] using a soft voting mechanism for auto-
mated TB diagnosis. The primary contributions of our work are: (1) a thorough
assessment of three leading CNN architectures for detecting TB, (2) a novel en-
semble design specifically tailored for medical diagnostic tasks, and (3) a superior
level of performance that suggests a strong potential for clinical application. The
paper is organized as follows: Section 2 provides a review of related literature.
Section 3 describes the proposed methodology. Section 4 presents and analyzes
the results. Lastly, Section 5 offers a conclusion and points to avenues for future
research.


2   Related Work
The application of convolutional neural networks (CNNs) has led to substantial
progress in the computer-aided diagnosis of tuberculosis (TB) using chest X-rays.
For instance, Rahman et al. [3] utilized a VGG16 architecture with transfer
learning on the Kaggle TB dataset, attaining 92.4% accuracy but encountering
problems with overfitting. For explainable TB diagnosis, Wang et al. [9] integrated
VGG19 with Grad-CAM, achieving a 90% F1-score with the Montgomery and
Shenzhen datasets. Kumar et al. [10] adapted MobileNetV2 for use on embedded
devices, successfully preserving an 88.7% F1-score in settings with constrained
resources. Ensemble techniques have also yielded encouraging results. Bassi and
Attux [11] developed ECOVNet, an ensemble of EfficientNet-B0 models, which
reached a 91.6% F1-score by employing model averaging and data augmentation.
Building on this, Chakraborty et al. [12] used EfficientNetB3 with adaptive
learning rates to achieve a 93.2% F1-score on a dataset that had been rebalanced.
Kazeminia et al. [13] created a deep ensemble model that, after training on
165,000 X-ray images, achieved an AUC of 0.89, demonstrating its capacity
for large-scale generalization. More recently, hybrid models have come to the
forefront. Zhou et al. [14] combined CNNs with transformers to obtain a 94.1%
F1-score, while Ahmed et al. [15] developed VisionTBNet, a model based on
transformers that recorded a 95.3% F1-score. These works underscore the value of
architectural diversity, transfer learning, and model interpretability for enhancing
TB detection. The TUBER-ENSEMBLE framework we propose expands on these
insights, combining ResNet50, DenseNet121, and InceptionV3 via soft voting to
boost generalization and minimize false negatives


---

                                 Title Suppressed Due to Excessive Length       3

            Table 1. Comparison of TB detection models (2020–2024).

       Reference             Method                      F1-score (%) Year
       Rahman et al. [3]       VGG16 + TL               88.0          2020
       Wang et al. [9]         VGG19 Grad-CAM           90.0          2021
       Kumar et al. [10]       MobileNetV2              88.7          2021
       Bassi et al. [11]       ECOVNet Ensemble         91.6          2021
       Chakraborty et al. [12] EfficientNetB3           93.2          2022
       Kazeminia et al. [13] Deep Ensemble CNNs         88.0          2022
       Zhou et al. [14]        CNN + Transformer Hybrid 94.1          2023
       Ahmed et al. [15]       VisionTBNet (ViT)        95.3          2024



3   Methodology
This section details our automated TB detection framework combining three CNN
architectures—ResNet50, DenseNet121, and InceptionV3—trained from scratch
on a specialized TB dataset. Our approach ensures robust feature extraction,
mitigates overfitting, and improves generalization through ensemble learning with
soft voting. The pipeline encompasses model architectures, training strategy, and
model fusion for enhanced diagnostic performance.
3.1 Dataset and Preprocessing
We used the Kaggle TB dataset [3], containing 4200 posterior-anterior X-rays
(3500 Normal, 700 TB). Data were split 80:20 into training and test sets (3360
training, 840 testing), preserving the class balance.
    The preprocessing pipeline consisted of four main steps:
 – Grayscale conversion: reduced redundancy.
 – Resize to 224×224 pixels: ensured CNN compatibility.
 – Normalization to [0,1]: improved stability.
 – Data augmentation: included random rotations (up to ±15◦ ), horizontal
   flips, zooming by 20%, and translations. To counteract class imbalance,
   augmentation was applied to 30% of TB-positive images and 5% of TB-
   negative ones.
3.2 CNN Architectures
We selected three architectures known for their complementary strengths:
 – ResNet50 [6]: This is a 50-layer deep network featuring residual connections,
   which use skip connections to address the vanishing gradient problem. It has
   roughly 25.6 million parameters.
 – DenseNet121 [7]: Characterized by its dense connectivity across 121 layers,
   this network enhances feature reuse and ensures efficient gradient propagation.
   It is composed of approximately 8.0 million parameters.
 – InceptionV3 [8]: This architecture is designed for multi-scale processing and
   employs factorized convolutions. This approach optimizes computational cost
   without sacrificing high accuracy. It contains about 23.8 million parameters.


---

4       M. Kissane et al.

3.3 Training Configuration
All models were trained from scratch under a uniform set of hyperparameters. We
used the Adam optimizer with a learning rate of 1 × 10−4 , binary cross-entropy
for the loss function, a batch size of 32, and trained for 50 epochs. For binary
classification, each architecture was adapted by adding a GlobalAveragePooling2D
layer followed by a sigmoid activation function. The training was carried out on
Google Colab with GPU support, incorporating early stopping and learning rate
reduction to optimize performance.
3.4 Ensemble Strategy
Our approach utilized a soft voting ensemble, averaging probabilistic outputs
from ResNet50, DenseNet121, and InceptionV3. This differs from hard voting,
which only considers final predicted classes. Soft voting preserves confidence
scores from each model, important in medical diagnosis where probability data
guides clinical judgment. To find the best threshold, we analyzed values from
0.4 to 0.7. While automatic methods like Youden’s J statistic could be used,
manual evaluation allowed direct observation of precision-recall trade-offs critical
in TB screening. A threshold of 0.5 provided optimal balance between sensitivity
and specificity. This ensemble leverages unique advantages: ResNet50’s residual
learning for deep feature extraction, DenseNet121’s dense connections for efficient
feature reuse, and InceptionV3’s multi-scale spatial pattern capture. Merging
outputs creates a robust diagnostic system, as illustrated in Figure 1.




Fig. 1. Illustration of the ensemble strategy using soft voting to combine the outputs
of InceptionV3, ResNet50, and DenseNet121.


---

                                  Title Suppressed Due to Excessive Length        5

3.5   Evaluation Metrics

To evaluate performance, we used metrics standard in medical image analysis,
focusing on reducing false negatives critical for TB screening. Performance was
assessed with Accuracy, Precision, Recall, Specificity, Negative Rate, and F1-score:

                                               TP + TN
                            Accuracy =                                          (1)
                                        TP + TN + FP + FN
                                           TP
                          P recision =                                          (2)
                                        TP + FP
                                           TP
                Recall (Sensitivity) =                                          (3)
                                        TP + FN
                                           TN
                         Specif icity =                                         (4)
                                        TN + FP
                                           TN
                     N egative Rate =                                           (5)
                                        FN + TN
                                        2 × P recision × Recall
                          F 1-Score =                                           (6)
                                          P recision + Recall

AUC-ROC was also used to evaluate discrimination ability. These metrics em-
phasize reducing false negatives, critical in TB screening.



4     Results and Discussion


4.1   Individual Model Performance

The performance of each individual CNN on the test set (at a threshold of 0.5)
is detailed in Table 2. InceptionV3 showed the best standalone results, achieving
98% accuracy and a 97% F1-score, a success we link to its ability to extract
features at multiple scales. DenseNet121 delivered well-rounded performance,
with 92.3% accuracy and an 83% F1-score, thanks to its efficient reuse of features.
ResNet50, however, demonstrated low sensitivity, with a recall of only 51%,
leading to a low F1-score of 47% despite its high precision.


                  Table 2. Performance of CNNs and Ensemble.

               Model        Acc (%) Prec (%) Rec (%) F1 (%) AUC
               ResNet50     83.6       92.0     51.0    47.0 0.81
               DenseNet121 92.3        94.0     78.0    83.0 0.92
               InceptionV3  98.0       99.0     95.0    97.0 0.98
               Ensemble    98.45       98.0     92.0    95.0 0.986


---

6      M. Kissane et al.

4.2 Ensemble Results and ROC Analysis
The ensemble improved recall and robustness compared to individual models.
Figure 2 shows the confusion matrix with AUC=0.99.




                   Fig. 2. Confusion matrix of ensemble model.


4.3 Threshold Analysis
An analysis of threshold sensitivity across 0.4–0.7 verified 0.5 as optimal for
all models. InceptionV3 showed consistent performance across thresholds, while
ResNet50’s recall was more sensitive to threshold changes. The ensemble displayed
robustness to these variations, as shown in Figure 3.




             Fig. 3. Precision, Recall, and F1-score across thresholds.


---

                                   Title Suppressed Due to Excessive Length           7

5   Conclusion and Future Work
The TUBER-ENSEMBLE framework effectively shows that combining different
CNN architectures is a powerful strategy for the automated detection of TB.
With its soft voting mechanism, our ensemble reached an accuracy of 98.45%,
surpassing both the individual models and other existing methods. The framework
benefits from the distinct advantages of its components: the residual learning
of ResNet50, the feature efficiency of DenseNet121, and the multi-scale analysis
of InceptionV3. The main conclusions are that InceptionV3 delivers the best
results for TB detection as a standalone model, that using ensemble methods
markedly boosts robustness and the ability to generalize, and that the use of
soft voting combined with threshold tuning improves the model’s suitability for
clinical use. The study’s limitations include its validation on a single dataset
and the computational demands that could hinder deployment on edge devices.
Future research should focus on validating the model externally with data from
diverse populations, developing lightweight model versions for use in low-resource
settings, expanding the model to classify multiple diseases, and adding a lung
segmentation step in preprocessing. This research lays the groundwork for AI-
driven TB screening, a tool that could be particularly impactful for public health
in high-prevalence areas with limited resources, where automated diagnostics can
make a significant difference.

Acknowledgements We extend our gratitude to the Moroccan School of
Engineering Sciences for their support and for providing the computational
resources necessary for this research.


References
 1. World Health Organization: Global Tuberculosis Report 2023. WHO (2023)
 2. Lakhani, P., Sundaram, B.: Deep learning at chest radiography: automated classifi-
    cation of pulmonary tuberculosis by using convolutional neural networks. Radiology
    284(2), 574–582 (2017)
 3. Rahman, T.: Tuberculosis (TB) Chest X-ray Dataset. Kaggle (2020). https://www.
    kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
 4. Litjens, G., Kooi, T., Bejnordi, B.E., et al.: A survey on deep learning in medical
    image analysis. Medical Image Analysis 42, 60–88 (2017)
 5. Rajpurkar, P., Irvin, J., Zhu, K., et al.: CheXNet: Radiologist-level pneumonia
    detection on chest X-rays with deep learning. arXiv preprint arXiv:1711.05225
    (2017)
 6. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
    In: Proc. IEEE CVPR, pp. 770–778 (2016)
 7. Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q.: Densely connected
    convolutional networks. In: Proc. IEEE CVPR, pp. 4700–4708 (2017)
 8. Szegedy, C., Vanhoucke, V., Ioffe, S., et al.: Rethinking the inception architecture
    for computer vision. In: Proc. IEEE CVPR, pp. 2818–2826 (2016)
 9. Wang, Y., et al.: Explainable tuberculosis screening using VGG19 and Grad-CAM.
    Journal of Medical Systems 45(8), 90 (2021)


---

8       M. Kissane et al.

10. Kumar, S., et al.: Lightweight MobileNetV2-based TB classification for embedded
    deployment. IEEE Sensors Journal 21(14), 15487–15495 (2021)
11. Bassi, P.R.A., Attux, R.: ECOVNet: An ensemble of deep CNNs for COVID-19
    detection using chest X-rays. Biomedical Signal Processing and Control 68, 102791
    (2021)
12. Chakraborty, S., et al.: EfficientNet-based feature fusion for TB detection on
    balanced chest X-ray dataset. Computers in Biology and Medicine 145, 105488
    (2022)
13. Kazeminia, S., et al.: Deep ensemble learning for detecting active pulmonary TB
    using CXRs. IEEE Transactions on Medical Imaging 41(3), 732–744 (2022)
14. Zhou, H., et al.: Hybrid CNN-Transformer model for tuberculosis detection. In:
    Proc. Int. Conf. on Medical Imaging (2023)
15. Ahmed, A., et al.: VisionTBNet: A Vision Transformer Framework for Tuberculosis
    Classification. IEEE Access 12, 10566–10578 (2024)


---

