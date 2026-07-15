---
source_pdf: 662647.pdf
pages: 11
converted_with: pdftotext -layout
---

# 662647.pdf

> Auto-converted from PDF. Prefer this file over the PDF for AI reading.

---

       Multi-Category Semantic Segmentation on
        CT-Scans Using Deep Neural Networks

    Imad Issame2 , Amine Zeguendry1,2[0000−0002−5864−5667] , and Mohammed
                              Amine Agoumi2
1
    Laboratory of Computer Systems and Intelligent Systems, Cadi Ayyad University,
                   UCA, Faculty of Sciences Marrakech, Morocco
      2
        LAMIGEP, EMSI, Moroccan School of Engineering Marrakech, Morocco
                              a.zeguendry@emsi.ma




       Abstract. Semantic segmentation of medical images is essential for en-
       hancing computer-aided diagnosis and treatment planning. This work
       addresses the multi-category semantic segmentation of CT images, tar-
       geting five categories: Background, Liver, Kidney, Spleen, and Pancreas.
       Two state-of-the-art models, UNet++ and DeepLabv3, pretrained on
       the ImageNet dataset, are employed to benchmark performance. Addi-
       tionally, a custom-designed model is implemented, incorporating a UNet
       backbone with residual blocks in the encoder and advanced attention
       mechanisms such as attention gates and squeeze-and-excitation blocks in
       the decoder. The models are evaluated using precision, recall, F1-score,
       IoU score, and Dice score metrics for each organ. Comparative analysis
       reveals the strengths and limitations of each approach, with the custom
       model showcasing significant improvements in capturing finer anatomical
       structures. The results contribute to the ongoing advancement of deep
       learning-based segmentation techniques for medical imaging.

       Keywords: Medical Image Segmentation · Semantic Segmentation ·
       UNet++ · DeepLabv3 · Residual Blocks · Attention Gates · Squeeze-
       and-Excitation · Abdominal CT · Deep Learning · Multi-Category Seg-
       mentation


1     Introduction
Medical image segmentation is an important task in healthcare, helping doc-
tors analyze images faster and more accurately. It is widely used for identifying
organs, detecting diseases, and planning treatments. Abdominal organ segmen-
tation in CT scans is particularly challenging because organs vary in shape, size,
and position among patients.
    Deep learning models have brought significant improvements to this field.
Popular models like U-Net [1], UNet++, DeepLabv3, and self-configuring frame-
works such as nnU-Net [3] are known for their high performance in image seg-
mentation. These models use advanced designs to process complex features and
provide accurate results. However, some organs, like the pancreas, remain diffi-
cult to segment because of their small size or unclear boundaries.


---

2       I. Issame et al.

    In this project, we focus on segmenting five categories from CT scans: Back-
ground, Liver, Kidney, Spleen, and Pancreas. Two state-of-the-art models, UNet++
and DeepLabv3, pretrained on the ImageNet dataset, are used as benchmarks.
Additionally, we develop a custom model based on a UNet structure. This model
includes residual blocks in the encoder and attention mechanisms, such as at-
tention gates and squeeze-and-excitation blocks, in the decoder. These additions
aim to improve the segmentation, especially for smaller or less visible organs.
    The performance of the models is measured using precision, recall, F1-score,
Intersection over Union (IoU), and Dice score. The results show how each model
performs and highlight the strengths of the custom model. This work provides
valuable information on the use of deep learning for medical image segmentation
and contributes to the development of better tools for healthcare applications.
2     Methods
In this section, we detail the methodologies used to address the semantic seg-
mentation of abdominal CT images. We first describe the two state-of-the-art
models, DeepLabv3 and UNet++, which serve as benchmarks for performance.
For each model, we explore the architecture, key components, and their limita-
tions. Following this, we introduce a self-made model designed to overcome some
of the identified limitations, focusing on small and less distinct anatomical struc-
tures such as the pancreas. The custom model incorporates novel architectural
elements to improve segmentation performance and precision.
2.1   State-of-the-Art Models
DeepLabv3/DeepLabv3+ [5, 2] is a leading semantic segmentation model that
uses atrous convolutions and an Atrous Spatial Pyramid Pooling (ASPP) module
to capture multi-scale features. It employs a ResNet-50 backbone pretrained on
ImageNet, ensuring strong feature extraction and an effective trade-off between
spatial resolution and computational efficiency. However, it struggles with small
object segmentation, is computationally expensive, and depends heavily on the
quality of the pretrained backbone.
    UNet++ [6] extends the original U-Net by introducing nested dense skip
connections to enhance feature refinement and gradient flow. Implemented with
ResNet-50 and EfficientNet-B7 backbones (both pretrained on ImageNet), it
achieves high segmentation accuracy but at the cost of increased memory usage,
longer training time, and risk of overfitting on small datasets. Despite these
limitations, UNet++ remains a strong candidate for medical image segmentation
tasks requiring high precision.
2.2 Proposed Model
The proposed model is an advanced version of the U-Net architecture, enhanced
with residual blocks, attention gates, and squeeze-and-excitation (SE) blocks
(Figure 1).
    This combination aims to improve the model’s ability to capture intricate
features and focus on the most relevant parts of the input data.
    Residual blocks (Figure 2) help in training deeper networks by allowing gra-
dients to flow more easily, while attention gates enable the model to concentrate


---

                                       Semantic Segmentation on CT Scans          3

on important regions, enhancing feature representation. Additionally, SE blocks
(Figure 3) adaptively recalibrate channel-wise feature responses, further refining
the learned features. Together, these components make the model more robust
and effective for tasks such as image segmentation.




Fig. 1: Proposed model architecture (residual encoder, attention-enhanced skips, SE-
based decoder).




Residual Blocks Residual blocks [8] are essential components designed to over-
come the vanishing gradient problem in deep neural networks. In this model,
each residual block consists of a sequence of operations that include a 3x3 con-
volutional layer, batch normalization, and a ReLU activation function (Figure
2).
    Specifically, the block begins with a 3x3 convolution that captures spatial fea-
tures from the input. This is followed by batch normalization, which stabilizes
and accelerates the training process by normalizing the output of the convolu-
tional layer. The ReLU activation introduces non-linearity, enabling the network
to learn complex patterns.
    A key feature of these residual blocks is the inclusion of a shortcut or "resid-
ual" connection. Mathematically, if the input to the block is x and the transfor-
mation applied by the convolutional layers is F (x), the output of the residual
block is given by:




                               Output = F (x) + x


---

4      I. Issame et al.




                Fig. 2: Residual block used in the proposed model.


    This residual connection allows the network to pass the input directly to the
output, facilitating better gradient flow during training. By adding the input x
to the transformed output F (x), the network can effectively learn residual func-
tions. This approach makes it easier to train deeper models without experiencing
performance degradation, as it mitigates the risk of vanishing gradients.
    In the context of the proposed U-Net architecture, the residual blocks enable
the construction of a deeper network by ensuring that gradients can flow more
freely through the layers. The use of 3 × 3 convolutions within these blocks
enhances the model’s ability to extract detailed spatial features, which is crucial
for tasks like image segmentation.
Attention Block Inspired by the attention mechanisms introduced in [7], the
Attention Block is designed to enhance the U-Net architecture by enabling the
model to focus on the most relevant features during the decoding process.
    Attention gates serve as dynamic filters that selectively emphasize important
spatial regions in the feature maps while suppressing irrelevant information. This
selective focus improves the model’s ability to capture essential details, leading
to more accurate and efficient feature representations.
    Mathematically, the attention gate operates by combining a gating signal g
with the input feature map x from the encoder. The process can be described
by the following equation:
                                                         
                          ψ = σ Wψ · ReLU(Wg g + Wx x)

Here, σ represents the sigmoid activation function, which normalizes the atten-
tion weights ψ between 0 and 1. The terms Wg and Wx are learnable weight
matrices that project the gating signal g and the input feature map x into a


---

                                        Semantic Segmentation on CT Scans       5

common feature space. The ReLU activation introduces non-linearity, allowing
the model to capture complex relationships between g and x.
   The resulting attention weights ψ are then used to modulate the input feature
map x, effectively highlighting important regions and diminishing less relevant
ones:



                                     x̃ = x · ψ




   The primary reason for incorporating attention gates into the model is to
enhance the network’s ability to focus on relevant spatial regions, thereby im-
proving feature selection and representation. By refining the skip connections
with attention weights, the model ensures that only the most pertinent features
from the encoder are passed to the decoder.
Squeeze-and-Excitation Block Inspired by the Squeeze-and-Excitation (SE)
mechanism introduced in [9], the SE Block enhances the U-Net architecture by
adaptively recalibrating channel-wise feature responses (Figure 3). The SE block
operates in two steps: squeeze and excitation. During the squeeze phase, global
average pooling aggregates spatial information for each channel, producing a
channel descriptor:



                                          H X W
                                     1   X
                           sc =                  xc,i,j
                                   H × W i=1 j=1




In the excitation phase, these descriptors pass through two fully connected layers
with a ReLU activation and a sigmoid function to generate scaling factors:



                                     1   X
                            sc =             xc,i,j ,
                                   H × W i,j



The original feature map is then scaled by these factors:



                                    x̃c = xc × sc


---

6       I. Issame et al.




              Fig. 3: Squeeze-and-Excitation block used in the decoder.




    The primary purpose of SE blocks is to model and emphasize the most in-
formative channels, improving the network’s ability to focus on relevant features
while suppressing less important ones. In this model, SE blocks are integrated
into the upsampling blocks of the U-Net, allowing for refined feature represen-
tation during the decoding process.
3     Experiments
3.1   Dataset and Preprocessing
The dataset is based on AbdomenCT-1k [10], containing 892 labeled training
samples and 143 unlabeled test samples of abdominal CT images. Since test
labels were unavailable, the training set was split into 713 training and 179
validation samples. All images were resized to 224 × 224 pixels, and training was
conducted in mini-batches of size 32. Data loaders were verified for consistency,
producing tensors of size (32, 1, 224, 224) for images and (32, 224, 224) for labels.
3.2   Experimental Settings
Evaluation Metrics: To evaluate the segmentation performance, we utilized
two widely recognized metrics: Intersection over Union (IoU) and Dice Coeffi-
cient. These metrics quantify the overlap between the predicted segmentation
mask and the ground truth, offering detailed insight into the model’s perfor-
mance for each class.


---

                                        Semantic Segmentation on CT Scans          7

   Intersection over Union (IoU) is defined as the ratio of the intersection be-
tween the predicted mask P and the ground truth mask T to their union:
                                          |P ∩ T |
                                  IoU =
                                          |P ∪ T |
Dice Coefficient emphasizes the overlap by considering the size of both predicted
and ground truth regions. It is calculated as:
                                          2|P ∩ T |
                                 Dice =
                                          |P | + |T |
This metric ranges from 0 to 1, with higher values indicating better segmentation
performance. The Dice coefficient is particularly useful for evaluating small or
imbalanced classes and was popularized for volumetric medical segmentation by
V-Net [4].
Loss Functions: To optimize the segmentation model, we experimented with
multiple loss functions tailored to different aspects of the task:

CrossEntropy Loss This loss function is the standard choice for classification
and segmentation tasks. For multi-class segmentation, the CrossEntropy loss is
defined as:
                                   1 X
                         LCE = −          yi,c log(ŷi,c )
                                  N i,c
Here, N is the total number of pixels, C is the number of classes, yi,c is the ground
truth probability for pixel i and class c, and ŷi,c is the predicted probability.

Dice Loss This loss function directly optimizes the Dice Coefficient, focusing on
the overlap between the predicted and ground truth regions:
                                            2|P ∩ T |
                            LDice = 1 −
                                          |P | + |T | + ϵ
The term ϵ is a small constant added to avoid division by zero.
     Each loss function was evaluated to assess its impact on segmentation per-
formance, particularly in balancing precision and recall across different classes.
Optimizer and Training Process: The training process utilized the Adam
optimizer with a fixed learning rate of 0.001. Adam was chosen due to its ability
to adaptively adjust the learning rate for each parameter, ensuring stable and
efficient convergence. The model was trained using a batch size of 32, with data
processed in mini-batches to balance memory usage and computational efficiency.
4   Results
The performance evaluation of different segmentation models UNet++, DeepLabv3,
and a Proposed U-Net was conducted on organ segmentation tasks involving
the Liver, Kidney, Spleen, and Pancreas. The models were assessed using F1
Score, Precision, Recall, Dice Coefficient, and Intersection over Union (IoU). The
experiments considered various configurations, including pretrained and non-
pretrained backbones and the use of different loss functions (Cross Entropy and
Dice Loss). The detailed results are presented in Table 1, Table 2, and Table 3.


---

8          I. Issame et al.

4.1     Overall Performance Across Models
As shown in Table 1, Unet++ demonstrates high performance across all organs,
especially when using a pretrained ResNet50 backbone with Cross Entropy loss.
The model achieves an F1 Score of 0.998 for the Liver, with both Precision and
Recall at 0.999, indicating excellent segmentation accuracy.



                              Table 1: Summary of Results for UNet++
heightModel                              Loss Function Class   F1 Score Precision Recall Dice IoU
                                                       Liver     0.998    0.999   0.998 0.998 0.997
                                                       Kidney    0.927    0.895   0.961 0.927 0.866
Unet++, Pretrained (ResNet50)            Dice Loss
                                                       Spleen    0.957    0.943   0.972 0.957 0.917
                                                       Pancreas 0.912     0.940   0.889 0.912 0.842
                                                       Liver     0.998    0.999   0.998 0.998 0.997
                                                       Kidney    0.893    0.869   0.919 0.893 0.811
Unet++, Non Pretrained (ResNet50)        Cross Entropy
                                                       Spleen    0.938    0.925   0.951 0.938 0.882
                                                       Pancreas 0.877     0.889   0.875 0.877 0.788
                                                       Liver     0.998    0.999   0.998 0.998 0.997
                                                       Kidney    0.988    0.977   0.999 0.988 0.977
Unet++, Pretrained (EfficientNet-B7)     Cross Entropy
                                                       Spleen    0.977    0.987   0.969 0.977 0.956
                                                       Pancreas 0.938     0.933   0.943 0.938 0.882
                                                       Liver     0.983    0.990   0.978 0.983 0.967
                                                       Kidney    0.980    0.987   0.974 0.980 0.961
Unet++, Non Pretrained (EfficientNet-B7) Cross Entropy
                                                       Spleen    0.431    0.500   0.375 0.431 0.276
                                                       Pancreas 0.929     0.910   0.949 0.929 0.867




    Similarly, DeepLabv3 exhibits strong performance, particularly with a pre-
trained backbone and cross-entropy loss, as detailed in Table 2. The model at-
tains an F1 Score of 0.997 for the Liver, with a Precision of 0.996 and a Recall
of 0.999.



                            Table 2: Summary of Results for DeepLabv3
heightModel               Loss Function Class   F1 Score Precision Recall Dice IoU
                                        Liver    0.997    0.996 0.996 0.997 0.994
                                        Kidney   0.974    0.996 0.951 0.974 0.950
DeepLabv3, Pretrained     Cross Entropy
                                        Spleen   0.949    0.925 0.975 0.949 0.905
                                        Pancreas 0.967    0.979 0.956 0.967 0.936
                                        Liver    0.997    0.998 0.997 0.997 0.995
                                        Kidney   0.973    0.956 0.991 0.973 0.948
DeepLabv3, Non Pretrained Cross Entropy
                                        Spleen   0.946    0.943 0.949 0.946 0.897
                                        Pancreas 0.964    0.964 0.964 0.964 0.931
                                        Liver    0.996    0.997 0.994 0.996 0.993
                                        Kidney   0.926    0.901 0.952 0.926 0.865
DeepLabv3, Pretrained     Dice Loss
                                        Spleen   0.940    0.944 0.935 0.940 0.889
                                        Pancreas 0.954    0.984 0.927 0.954 0.913
                                        Liver    0.996    0.997 0.994 0.996 0.993
                                        Kidney   0.963    0.937 0.990 0.963 0.930
DeepLabv3, Non Pretrained Dice Loss
                                        Spleen   0.944    0.945 0.944 0.944 0.893
                                        Pancreas 0.956    0.865 0.956 0.956 0.914




    The Proposed Unet model achieves the highest scores in several metrics, as
seen in Table 3. For instance, it records an F1 Score of 0.999 for the Liver with
both Precision and Dice Coefficient at 0.999 and 0.997 respectively, highlighting
its superior effectiveness in organ segmentation tasks.


---

                                         Semantic Segmentation on CT Scans       9


                 Table 3: Summary of Results for Proposed Model
heightModel   Loss          Class   F1 Score Precision Recall Dice IoU
                            Liver    0.998    0.999 0.998 0.998 0.997
                            Kidney   0.987    0.980 0.994 0.987 0.974
Proposed Unet Cross Entropy
                            Spleen   0.979    0.969 0.990 0.979 0.960
                            Pancreas 0.943    0.952 0.934 0.943 0.892
                            Liver    0.997    0.995 0.999 0.997 0.995
                            Kidney   0.982    0.973 0.992 0.982 0.964
Proposed Unet Dice Loss
                            Spleen   0.974    0.969 0.980 0.974 0.949
                            Pancreas 0.942    0.989 0.902 0.942 0.891


4.2   Impact Of Pretraining
Pretrained models consistently outperform their non-pretrained counterparts. In
Table I, the Unet++ with a pretrained ResNet50 backbone and Cross Entropy
loss surpasses the non-pretrained version, achieving higher F1 Scores and IoU
values for most organs. DeepLabv3 with a pretrained backbone (Table 2) also
shows better performance metrics compared to the non-pretrained version, par-
ticularly in the segmentation of the Kidney and Pancreas. Non-pretrained models
tend to have lower performance, especially in challenging classes like the Spleen
and Pancreas. For example, the Unet++ with a non-pretrained EfficientNet-B7
backbone (Table 1) shows a significant drop in performance for the Spleen, with
an F1 Score of 0.431, indicating difficulty in accurately segmenting this organ
without pretraining.
4.3   Impact Of Loss Functions
Cross Entropy Loss Generally leads to better performance across most models
and classes. The Proposed Unet with Cross Entropy loss consistently achieves
the highest scores, such as an F1 Score of 0.999 for the Liver and 0.987 for the
Kidney. DeepLabv3 with Cross Entropy loss also exhibits strong performance,
especially when pretrained.
Dice Loss While Dice Loss is specifically designed for segmentation tasks, it
does not consistently outperform Cross Entropy loss in this evaluation. Some
models show decreased performance with Dice Loss. For instance, DeepLabv3
trained with Dice Loss displays lower F1 Scores for the Spleen compared to when
trained with Cross Entropy loss.
4.4 Performance on Different Organs
Liver All models perform strongly. The top F1 is 0.998, attained by multiple
configurations, including UNet++ (various settings) and the Proposed U-Net
(Cross Entropy) (Tables 1, 3). DeepLabv3 (Pretrained, Cross Entropy) follows
closely at 0.997 (Table 2). Qualitative examples in Figure 4, Figure 5, and Figure
6 show precise boundaries with minimal errors.


Kidney UNet++ (Pretrained EfficientNet-B7, Cross Entropy) achieves
the highest F1 at 0.988 (Table 1), with the Proposed U-Net (Cross Entropy)
at 0.987 (Table 3). DeepLabv3 reaches up to 0.974 (Pretrained, Cross Entropy;
Table 2). Figures 4 and 5 illustrate clean delineations and few misclassifications.


---

10       I. Issame et al.

Spleen The Proposed U-Net (Cross Entropy) obtains the best F1 at 0.979
(Table 3). UNet++ (Pretrained EfficientNet-B7, Cross Entropy) achieves 0.977
(Table 1), while DeepLabv3 peaks at 0.949 (Pretrained, Cross Entropy; Table
2). Figure 5 shows accurate spleen segmentation for UNet++; Figure 4 shows
strong results for the Proposed U-Net.

Pancreas The most challenging organ. The best F1 is 0.967 with DeepLabv3
(Pretrained, Cross Entropy) (Table 2). The Proposed U-Net reaches 0.943 (Cross
Entropy; Table 3), and UNet++ (Pretrained EfficientNet-B7, Cross Entropy)
attains 0.938 (Table 1). Figures 4 and 6 show that DeepLabv3 and the Proposed
U-Net best capture the pancreas, though small inaccuracies remain.




    Fig. 4: Qualitative examples: CT segmentation produced by the proposed model.




         Fig. 5: Qualitative examples: CT segmentation produced by UNet++.




        Fig. 6: Qualitative examples: CT segmentation produced by DeepLabv3.


5     Limitations
Although the proposed U-Net shows strong performance, pancreas segmentation
remains challenging due to its small size and fuzzy boundaries. Another limita-
tion is the reliance on pretrained backbones, which may reduce generalizability
in low-resource scenarios. Future work should investigate organ-specific models,
domain adaptation strategies, and ensemble methods to improve robustness.


---

                                        Semantic Segmentation on CT Scans          11

6    Conclusion
This work benchmarked UNet++, DeepLabv3, and a custom U-Net on ab-
dominal organ segmentation from CT scans. The proposed U-Net consistently
achieved superior results, particularly for difficult organs. Clinically, improved
segmentation can facilitate accurate diagnosis, treatment planning, and surgical
navigation. Future directions include addressing pancreas segmentation chal-
lenges, exploring domain adaptation to diverse imaging protocols, and integrat-
ing the model into real-world computer-aided diagnosis pipelines.
References
 1. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional Networks for Biomed-
    ical Image Segmentation. In: MICCAI, pp. 234–241. Springer (2015)
 2. Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H.: Encoder-Decoder with
    Atrous Separable Convolution for Semantic Image Segmentation. In: ECCV, pp.
    801–818 (2018)
 3. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnU-Net: A
    Self-Adapting Framework for U-Net-Based Medical Image Segmentation. Nature
    Methods 18, 203–211 (2021)
 4. Milletari, F., Navab, N., Ahmadi, S.A.: V-Net: Fully Convolutional Neural Net-
    works for Volumetric Medical Image Segmentation. In: 3DV, pp. 565–571 (2016)
 5. Chen, L.C., Papandreou, G., Schroff, F., Adam, H.: Rethinking Atrous Convolution
    for Semantic Image Segmentation. arXiv:1706.05587 (2017)
 6. Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., Liang, J.: UNet++: A Nested U-
    Net Architecture for Medical Image Segmentation. In: Deep Learning in Medical
    Image Analysis and Multimodal Learning for Clinical Decision Support, pp. 3–11.
    Springer (2018)
 7. Oktay, O., Schlemper, J., Le Folgoc, L., et al.: Attention U-Net: Learning Where
    to Look for the Pancreas. arXiv:1804.03999 (2018)
 8. He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition.
    In: CVPR, pp. 770–778 (2016)
 9. Hu, J., Shen, L., Sun, G.: Squeeze-and-Excitation Networks. In: CVPR, pp. 7132–
    7141 (2018)
10. Ma, J., et al.: AbdomenCT-1K: Is Abdominal Organ Segmentation a Solved Prob-
    lem? IEEE Transactions on Pattern Analysis and Machine Intelligence 44(10),
    6695–6714 (2021)


---

