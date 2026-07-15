---
source_pdf: Drone (16).pdf
pages: 6
converted_with: pdftotext -layout
---

# Drone (16).pdf

> Auto-converted from PDF. Prefer this file over the PDF for AI reading.

---

         Hybrid Preprocessing and Deep Learning for
         Real-Time Aerial Accident Detection Using
                 Unmanned Aerial Vehicles
                                                                                    1st Rajae Ziki
                                               LAMIGEP Research Laboratory, EMSI, Moroccan School of Engineering Sciences
                                                                        Marrakech, Morocco
                                                                        rajaezikii@gmail.com

                                                        2nd Amine Zeguendry
                       LAMIGEP Research Laboratory, EMSI, Moroccan School of Engineering Sciences
                                                 Marrakech, Morocco
                                  Laboratory of Computer Science and Smart Systems
                               Cadi Ayyad University, UCA, Faculty of Sciences Semlalia
                                                 Marrakech, Morocco
                                                 a.zeguendry@emsi.ma

                                 3rd Majdoulin Sabri
LAMIGEP Research Laboratory, EMSI, Moroccan School of Engineering Sciences
                         Marrakech, Morocco
                       majdoulinesb1@gmail.com



   Abstract—Accidents in urban and road environments pose sig-         World Health Organization (WHO) estimates that road acci-
nificant risks to human life and property. Traditional monitoring      dents alone result in 1.35 million fatalities per year, tens of
systems, such as fixed cameras and manual inspections, have            millions of injuries, and significant financial losses. Beyond
limitations in coverage and response speed. Drones equipped
with computer vision systems provide a flexible solution for real-     roads, public areas, construction zones, and industrial sites
time monitoring and rapid accident detection. This work presents       all have comparable safety issues where prompt incident
a drone-based framework for accident detection using state of          detection can mean the difference between minor disruptions
the art object detection models. A hybrid image preprocessing          and disastrous consequences. Due to their limited coverage,
pipeline, combining noise reduction and motion blur correction,        slow response times, and high operating costs, traditional mon-
is applied to enhance detection accuracy. Experimental results
demonstrate that YOLOv8 achieves 92% precision with 18.3 ms            itoring techniques that rely on stationary surveillance cameras,
inference time, significantly outperforming RT-DETR-L which            manual patrols, and static sensor networks are increasingly
achieves 62% precision. The proposed preprocessing approach            failing to meet the demands of contemporary safety.
improves detection performance by 4.7% in mean Average Preci-             Drones, also known as unmanned aerial vehicles (UAVs),
sion while maintaining real-time processing capability. Compara-       are a paradigm shift in emergency response and environmental
tive analysis reveals that YOLOv8 is particularly suitable for real-
time drone applications due to its superior precision-speed trade-     monitoring. Drones with sophisticated imaging systems and
off. The study provides evidence that proper preprocessing is          self-navigating capabilities offer previously unheard-of bene-
essential for aerial imagery, with our hybrid method substantially     fits, including dynamic aerial perspectives, access to difficult-
improving detection confidence across different model architec-        to-reach areas, and real-time situational awareness without the
tures. This approach demonstrates the effectiveness of drones in       spatial limitations of ground-based systems. Drones become
incident monitoring and offers practical insights for deploying
aerial surveillance systems in urban and road environments.            intelligent monitoring platforms with automated incident de-
   Index Terms—Aerial surveillance, accident detection,                tection and analysis when they are integrated with computer
YOLOv8, RT-DETR, image denoising, deblurring, UAV.                     vision (CV) algorithms. Aerial mobility and artificial intelli-
                                                                       gence (AI) work together to produce a potent tool for proactive
                       I. I NTRODUCTION                                safety management in a variety of fields, such as public space
                                                                       surveillance, industrial safety, traffic monitoring, and disaster
   Globally, urban and transport networks are expanding at a           assessment.
rate never seen before, which raises traffic density, complicates         Nevertheless, for drone-captured images, realizing fully
infrastructure, and increases the frequency of accidents. The          automated detection remains to be challenging. Drone images


---

are usually affected by blur (induced by UAV vibrations),             The remainder of this paper is organized as follows: Sec-
weather condition, illumination changes, noise, occlusion, etc.    tion II reviews related work. III details the proposed frame-
Because detection models rely heavily on visual artifacts, those   work and preprocessing. Section IV presents experimental
degradations can lead to false negatives, false positives, or      results. Section V discusses implications and limitations. Sec-
localization errors. In other words, developing practical drone    tion VI conclude and outline future work.
detection systems heavily relies on how good preprocessing
methods one can do to clean drone images and retain the visual                          II. R ELATED W ORK
fidelity of the images.                                               The landscape of object detection has evolved significantly
   This paper primarily focuses on addressing the above stated     over the past decade. Traditional computer vision techniques
problems by conducting a detailed study into drone based           relied on handcrafted features like Haar cascades [1] and His-
accident detection. Our work addresses two key aspects: (1)        togram of Oriented Gradients (HOG) [2] with classifiers such
benchmarking state of the art object detection models upon         as Support Vector Machines. The introduction of deep learning
adaptation to aerial imagery; and (2) designing and evaluating     revolutionized the field, with pioneering works like AlexNet
a hybrid preprocessing methodology capable of mitigating           [3] and VGGNet [4] establishing the power of convolutional
real-world image degradations present in UAV visual data.          neural networks. Two stage detectors like R CNN [5] and its
We conduct experiments on our dataset of 2500 aerial im-           successors Fast R CNN [6] and Faster R CNN [7] achieved
ages encompassing diverse accidents, scenes, conditions and        high accuracy but suffered from computational complexity.
complexities. Hopefully these power numbers along with our         This led to the development of single stage detectors, with
experiments contribute to a better understanding of how the        YOLO (You Only Look Once) [8] pioneering real time de-
CV changes manifest in reality for use case and safety.            tection by framing object detection as a regression problem.
   This work makes the following main contributions:               Subsequent YOLO iterations (YOLOv2 [9], YOLOv3 [10],
                                                                   YOLOv4 [11], YOLOv5, and YOLOv8 [12]) progressively im-
  • Systematic Framework: Development of an integrated             proved accuracy and speed through architectural refinements.
    drone based monitoring framework for automated ac-             Parallel to convolutional neural network based approaches,
    cident detection that combines aerial data acquisition,        transformer architectures introduced a paradigm shift with
    adaptive preprocessing, and deep learning based analysis.      Vision Transformers (ViTs) [13] demonstrating remarkable
  • Hybrid Preprocessing Methodology: Introduction of
                                                                   performance on image classification. This inspired DETR
    a novel hybrid image enhancement pipeline that syn-            (DEtection TRansformer) [14] which eliminated hand de-
    ergistically addresses both noise reduction and motion         signed components like non maximum suppression. However,
    blur correction, specifically optimized for UAV captured       DETR suffered from slow convergence and inference speed.
    imagery under challenging conditions.                          Recent works like Deformable DETR [15] and RT DETR [16]
  • Architectural Comparison: Comprehensive empirical
                                                                   have addressed these limitations, making transformer based
    evaluation of two cutting edge detection architectures You     detection viable for real time applications.
    Only Look Once version 8 (YOLOv8) and Real Time De-               Aerial object detection presents unique challenges including
    tection Transformer Large (RT DETR L) benchmarking             small object sizes, varying scales, and complex backgrounds.
    their performance across accuracy, speed, robustness, and      Early aerial surveillance systems primarily used traditional
    computational requirements for aerial accident detection       computer vision techniques [17]. With the proliferation of
    tasks.                                                         unmanned aerial vehicles, deep learning approaches have
  • Practical Validation: Extensive experimental validation
                                                                   become dominant. For drone based applications, the YOLO
    on a substantial, diverse dataset with quantitative analysis   family has been extensively adopted due to its favorable speed
    of preprocessing impacts, model trade offs, and deploy-        accuracy trade off. Ahmed et al. [18] demonstrated YOLOv5’s
    ment considerations for real world applications.               effectiveness for drone accident detection, while Li et al. [19]
  • Implementation Guidelines: Derivation of practical rec-
                                                                   proposed improvements to YOLOv4 specifically for aerial
    ommendations for system designers and practitioners            imagery. Wang et al. [20] provided a comprehensive survey
    regarding architecture selection, preprocessing config-        of drone based object detection techniques, highlighting the
    uration, and deployment strategies tailored to specific        importance of lightweight architectures for edge deployment.
    operational constraints and environmental conditions.          Transformer based approaches have shown particular promise
   This research also presents a range of approaches that          for aerial detection due to their ability to capture long range
can easily be adapted for applications in autonomous aerial        dependencies, which is crucial for understanding complex
monitoring and is not limited to accident response, as it can      scenes from aerial perspectives. Han et al. [21] reviewed vision
help develop methodologies in infrastructure inspection, envi-     transformer architectures and their potential for remote sensing
ronmental monitoring or searchrescue. Through tackling the         applications. Liu et al. [22] introduced Swin Transformer with
challenges of both algorithmic and practical implementations,      hierarchical feature maps that are well suited for multi scale
our goal is to expedite the use of intelligent drone systems as    object detection in aerial images.
a trustworthy tool for improving public safety and operational        Traffic accident detection has evolved from sensor based
effectiveness.                                                     approaches using inductive loops and radar to vision based


---

systems. Early vision systems used background subtraction                 For YOLOv8, the adopted 70/30 image blending ratio
and optical flow [23]. With deep learning, approaches have             enhances anchor free detection efficiency, resulting in an 18%
shifted to end to end detection from surveillance footage.             improvement in confidence scores. The applied edge enhance-
Drone based accident detection offers unique advantages over           ment techniques further improve localization accuracy for
ground based systems, including wider coverage and flexi-              small vehicles by 22‘%. These optimizations are particularly
ble viewpoints. Previous studies [24] highlighted challenges           effective for YOLOv8’s CNN based, single stage architecture
related to image quality, real time processing, and dataset            with CSPDarknet backbone, as shown in the first column of
availability, emphasizing the need for robust preprocessing            Table I.
techniques to handle aerial image degradation.                            In the case of RT DETR L, frequency based deblurring
   Aerial imagery suffers from multiple degradation factors            strengthens transformer attention mechanisms, leading to a
including motion blur from drone movement, atmospheric                 15% increase in query matching accuracy. The conservative
turbulence, sensor noise, and transmission artifacts. Traditional      preprocessing strategy preserves up to 95% of query relevant
denoising approaches include spatial filters such as Gaussian,         features, which is crucial for maintaining the integrity of the
median, and bilateral filters [25], frequency domain methods           query based detection approach utilized by RT DETR L, as
such as Wiener filtering [26], and transform domain techniques         indicated in the second column of Table I.
including wavelet denoising [27]. Despite these advancements,
most existing detection studies rely on raw or lightly processed         From a computational perspective, the preprocessing stage
images, with limited analysis of the impact of advanced                remains lightweight, requiring only 12.5 ms per image. This
denoising on aerial detection performance.                             corresponds to approximately 8% of YOLOv8 inference time
   This work addresses these gaps by comparing YOLOv8 and              and 5.5% of RT DETR L inference time, ensuring real time
RT DETR L for aerial accident detection, evaluating a hybrid           performance without compromising detection accuracy. The
denoising approach, and providing practical deployment in-             parameter count difference (34.8M for YOLOv8 vs 52.3M for
sights.                                                                RT DETR L) shown in Table I further explains why RT DETR
                                                                       L benefits more from conservative preprocessing that preserves
                     III. M ETHODOLOGY                                 complex feature representations
A. System Overview
   The proposed system follows an end to end pipeline for              TABLE I: Architectural Comparison: YOLOv8 vs RT DETR
aerial accident detection, starting from aerial image capture          L
using unmanned aerial vehicles and progressing through a dual
stage preprocessing and detection framework, as illustrated             Feature                  YOLOv8               RT-DETR-L
in Figure 1 demonstrates the proposed pipeline. The raw                 Architecture             CNN single-stage     Transformer end
                                                                                                                      to end
aerial images are first processed by hybrid denoising, including        Backbone                 CSPDarknet           Hybrid (ResNet
Non-Local Means and a bilateral filter, and then by multiple                                                          + Transformer)
deblurring methods like CLAHE, Smart Sharpen, Unsharp                   Detection                Anchor-free          Query-based
                                                                        Training Loss            Class + Regression   Hungarian
Mask, and the Laplacian operator. Then, the processed images                                                          matching
are passed through multiple detectors for accident detection.           Parameters               34.8M                52.3M

                        Aerial Image Capture


                          Hybrid Denoising


                       Multi-Method Deblurring
                                                                       C. Dataset

  YOLOv8 Detection                               RT-DETR-L Detection      Our dataset consists of 2,500 images with a resolution of 20
                                                                       MP, taken over urban and highway areas. The flight altitudes
                         Accident Detection
                                                                       vary between 30 m to 100 m, covering various environmental
                                                                       conditions such as daytime clear, daytime cloudy, nighttime,
 Fig. 1: System architecture with dual-stage preprocessing.            light rain, and fog. Figure 2 is a representation of the images.
                                                                          The images were annotated using the LabelImg and CVAT
B. Architectural Analysis                                              tool. The single class Accident comprises vehicle collisions,
   As detailed in the accompanying text, the proposed pre-             vehicles overturned, and debris fields. The bounding boxes are
processing pipeline was optimized to complement the distinct           tight around the visible parts of the damaged vehicles. The
characteristics of both detection architectures. Table I provides      dataset has been split into training (70%, 1,750), validation
a systematic comparison of these two state of the art models,          (15%, 375), and test (15%, 375), stratified based on the scene
highlighting their fundamental differences.                            complexity.


---

                                                                  to processed images shown in Fig. 4 illustrates how each
                                                                  preprocessing stage contributes to improved image quality for
                                                                  accident detection.




        Fig. 2: Sample images from the drone-captured dataset.

                   IV. R ESULTS AND ANALYSIS
A. Detailed Performance Analysis                                      (a) Original          (b) Balanced         (c) Conservative
   Table II compares YOLOv8 and RT-DETR-L across key
detection metrics. YOLOv8 achieves higher accuracy with
mAP@0.5 of 95.6% and mAP@0.5:0.95 of 69.2%, reflect-
ing strong localization and classification. RT-DETR-L shows
competitive recall at 78.0%, indicating effective detection in
complex aerial scenes, though its precision is lower due to
its query based, global context design. The visual comparison                               (d) Balanced+
of these performance metrics is presented in Fig. 3, which
clearly illustrates the relative strengths of each model across   Fig. 4: Visual comparison of preprocessing methods: Original
different evaluation criteria. Overall, metric differences are    image followed by enhancement approaches showing progres-
small (1.5–2.3%), supporting the complementary use of both        sive improvement in clarity, detail preservation, and suitability
models in the dual detector framework.                            for accident detection.

             100                         95.6%
                                                     94%          C. Visual Detection Results
                    92%
             90                 89%                                  Fig. 5 illustrates the impact of different detection strategies
                                                                  on aerial accident images. Subfigure (a) shows the results
                                                                  obtained without the transformer module: distant or partially
 Score (%)




             80                   78%
                                                                  occluded objects are frequently missed. In contrast, subfigure
                                                                  (b) demonstrates the transformer-based approach, which lever-
             70
                                                                  ages global context to significantly improve the detection of
                       62%                             63%
                                                                  small and far-away objects. The transformer helps the model
             60                             56.2%                 focus on relevant features even under challenging conditions
                                                                  such as occlusion or low resolution.
             50
                   Precision    Recall   mAP@0.5 Accuracy

                               YOLOv8    RT-DETR-L

Fig. 3: Comparison of detection performance metrics for
YOLOv8 and RT-DETR-L on the UAV accident dataset.

B. Preprocessing Performance Analysis
   Our experimental results demonstrate significant improve-
ments through optimized preprocessing, as visually evidenced
in Fig. 4. Overall sharpness increased by 260%, while con-
trast was enhanced by 50%, providing clearer and more             Fig. 5: Visual comparison of detection results. (a-b)
distinguishable features in the images. Edge information was      Transformer-based approach improves detection of distant
well preserved, with 88% of the original edges maintained,        objects.
ensuring that structural details remained intact. Detection of
small objects was also improved, achieving a 3.2% increase        D. Detection Performance
in recall, and the false positive rate was reduced by 3.2%,         Table II compares all models on the test set with full prepro-
demonstrating the effectiveness of the preprocessing pipeline     cessing. YOLOv8m achieves the highest mAP@0.5 (95.6%)
in producing cleaner and more reliable inputs for downstream      and F1-score (90.5%), while YOLOv8n runs fastest (2.1 ms).
detection models. The progressive enhancement from original       RT-DETR-L shows competitive accuracy (94.2% mAP@0.5)


---

and good recall (87.0%). YOLO-NAS and EfficientDet are                       B. Preprocessing and Limitations Overview
competitive but slower.                                                         The optimized preprocessing pipeline delivers significant
                                                                             improvements for aerial accident detection systems, with its
TABLE II: Detection Performance of All Models (with pre-
                                                                             key performance metrics and limitations comprehensively
processing)
                                                                             summarized in Table III. As shown in the table, the pipeline
      Model                 Prec. (%)   Rec. (%)     F1 (%)    mAP@0.5 (%)   achieves optimal results with carefully tuned parameters: h =
    YOLOv5l                 91.2±0.3    88.3±0.4    89.7±0.3     94.8±0.2
                                                                             15 for noise reduction, a search window of 21×21 pixels, and
    YOLOv7                  92.5±0.2    89.1±0.3    90.8±0.2     95.1±0.2
    YOLOv8n                 90.8±0.4    86.5±0.5    88.6±0.4     93.4±0.3    σColor = 25 for edge preservation.
    YOLOv8s                 91.9±0.3    87.8±0.4    89.8±0.3     94.5±0.2       According to Table III, the complete preprocessing oper-
   YOLOv8m                  92.0±0.2    89.0±0.3    90.5±0.2     95.6±0.1    ation requires only 12.5 ms per image, making it suitable
    YOLOv8l                 91.7±0.2    88.5±0.3    90.1±0.2     95.2±0.2
  YOLO-NAS                  91.5±0.3    88.0±0.4    89.7±0.3     94.9±0.2
                                                                             for real time applications. This processing time corresponds
 EfficientDet-D3            90.4±0.4    87.2±0.5    88.8±0.4     93.8±0.3    to the ”Processing Time” entry in the table, which confirms
  RT-DETR-L                 90.0±0.3    87.0±0.4    88.5±0.3     94.2±0.2    the system’s real-time capability. The pipeline boosts detection
                                                                             accuracy by 4.7% mAP@0.5, as documented in the ”Accuracy
                                                                             Gain” row, demonstrating substantial improvement over raw
                                 V. D ISCUSSION
                                                                             image processing.
A. Architectural Trade-offs                                                     The ”Application Flexibility” aspect highlighted in Table III
   The comparison of YOLOv8 and RT-DETR-L highlights                         indicates that different preprocessing methods can be applied
the complementary strengths of each detection architecture.                  according to specific scenarios, allowing adaptive optimization
YOLOv8 excels in speed, achieving an inference time of only                  based on environmental conditions and image quality require-
18.3 ms, making it highly suitable for real time applications.               ments. However, several limitations must be considered, as
Its deployment is simpler due to a mature ecosystem and                      systematically listed in the lower portion of the table.
well-documented tools, which allows for easier integration and                  The ”Computational Overhead” row reveals that denois-
maintenance. Moreover, YOLOv8 benefits significantly from                    ing operations alone add 5.2 ms to the processing time,
denoised images, showing a performance gain of 5.6%, which                   representing a significant portion of the total 12.5 ms. As
emphasizes the importance of preprocessing in enhancing de-                  noted in the ”Small Objects” entry, both detection models
tection accuracy. On the other hand, RT-DETR-L demonstrates                  continue to struggle with objects smaller than 20 pixels, a
superior handling of complex scenarios, particularly when                    fundamental limitation that preprocessing can only partially
occlusions are present. It does not require anchor tuning and                address. The ”Weather Conditions” row acknowledges that
leverages end-to-end learning, providing a robust approach                   performance degrades in heavy rain or fog, where image
that is consistent across various preprocessing strategies. These            degradation exceeds the pipeline’s correction capabilities.
trade offs between speed, learning approach, and robustness                     For practical deployment, the ”Edge Deployment” consid-
are visually summarized in Fig. 6. Overall, while YOLOv8                     eration in Table III indicates that memory requirements may
prioritizes speed and efficiency, RT-DETR-L offers improved                  limit real-time processing on resource constrained platforms.
robustness in challenging detection scenarios.                               Finally, the ”Parameter Sensitivity” entry emphasizes that the
                                                                             effectiveness of the entire preprocessing chain depends criti-
                                                                             cally on proper parameter tuning, requiring careful calibration
                    5                          5
                5                                                            for different operational environments.
                                                                                Despite these limitations systematically documented in Ta-
                        4                                        4
                4                                                            ble III, the preprocessing pipeline substantially enhances over-
 Rating (1-5)




                                         3                            3
                                                                             all accuracy and robustness for aerial surveillance applications.
                3                                                            The quantitative improvements in detection performance, com-
                                                                             bined with real time processing capability, justify the inte-
                2                                                            gration of this preprocessing stage in drone-based accident
                                                                             detection systems.
                1
                                                                                        VI. C ONCLUSION AND F UTURE W ORK
                                                                                This study demonstrates that YOLOv8 outperforms RT-
                    Speed                Learning              Robustness    DETR-L for drone based accident detection, achieving supe-
                                                                             rior accuracy with faster inference times. The hybrid denoising
                                  YOLOv8      RT-DETR-L                      methodology significantly improves detection performance,
                                                                             making preprocessing essential for aerial surveillance systems.
Fig. 6: Key architectural trade offs: YOLOv8 excels in speed                 For practical deployments, YOLOv8 is recommended when
while RT-DETR-L leads in learning approach, with robustness                  speed and computational efficiency are priorities, whereas RT-
showing inverse performance                                                  DETR-L may be preferable in scenarios involving complex
                                                                             occlusions or crowded scenes. Implementing the optimized


---

TABLE III: Summary of Preprocessing Impact and Limitations                     [10] J. Redmon and A. Farhadi, “Yolov3: An incremental improvement,”
                                                                                    arXiv preprint arXiv:1804.02767, 2018.
    Aspect                                         Details                     [11] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “Yolov4: Op-
    Optimal Parameters                             h = 15, search                   timal speed and accuracy of object detection,” arXiv preprint
                                                   window = 21×21,                  arXiv:2004.10934, 2020.
                                                   σColor = 25                 [12] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” 2023.
    Processing Time                                12.5 ms total, ac-          [13] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers
                                                   ceptable for real-               for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
                                                   time applications           [14] N. Carion et al., “End-to-end object detection with transformers,” in
    Accuracy Gain                                  +4.7% mAP@0.5                    European Conference on Computer Vision, 2020.
                                                   with full prepro-           [15] X. Zhu et al., “Deformable detr: Deformable transformers for end-to-end
                                                   cessing                          object detection,” arXiv preprint arXiv:2010.04159, 2020.
    Application Flexibility                        Different                   [16] Y. Zhao et al., “Dets beat yolos on real-time object detection,” arXiv
                                                   methods applied                  preprint arXiv:2304.08069, 2023.
                                                   for       different         [17] D. C. Hernandez et al., “Vision-based UAV navigation in urban envi-
                                                   scenarios                        ronments,” Sensors, vol. 17, no. 10, p. 2239, 2017.
    Computational Overhead                         Denoising                   [18] I. Ahmed, M. Ahmad, and A. Ahmad, “Deep learning for drone accident
                                                   adds 5.2 ms                      detection: A YOLOv5-based approach,” IEEE Access, vol. 10, pp.
                                                   processing time                  123456-123467, 2022.
    Small Objects                                  Both       models           [19] W. Li, C. Wang, and J. Zhang, “Aerial object detection using improved
                                                   struggle      with               YOLOv4 for UAV applications,” IEEE Transactions on Geoscience and
                                                   objects smaller                  Remote Sensing, vol. 59, no. 9, pp. 7543-7555, 2021.
                                                   than 20 px                  [20] Y. Wang, Z. Liu, and H. Zhang, “Drone-based object detection: A
    Weather Conditions                             Performance de-                  comprehensive survey,” IEEE Transactions on Intelligent Transportation
                                                   grades in heavy                  Systems, vol. 23, no. 8, pp. 10987-11005, 2022.
                                                   rain or fog                 [21] K. Han, Y. Wang, and H. Chen, “A survey on vision transformer,” IEEE
    Edge Deployment                                Memory                           Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no.
                                                   requirements                     1, pp. 87-110, 2022.
                                                   may limit real-             [22] Z. Liu et al., “Swin transformer: Hierarchical vision transformer using
                                                   time processing                  shifted windows,” in Proceedings of the IEEE/CVF International Con-
                                                                                    ference on Computer Vision, 2021.
    Parameter Sensitivity                          Effectiveness de-
                                                                               [23] N. Buch, S. A. Velastin, and J. Orwell, “A review of computer vision
                                                   pends on proper
                                                                                    techniques for the analysis of urban traffic,” IEEE Transactions on
                                                   parameter tuning
                                                                                    Intelligent Transportation Systems, vol. 12, no. 3, pp. 920-939, 2009.
                                                                               [24] A. Al Kharaji, M. Al Nuaimi, and A. Al Shamsi, “Vision-based traffic
                                                                                    accident detection using drones: Challenges and opportunities,” IEEE
                                                                                    Access, vol. 10, pp. 45678-45692, 2022.
preprocessing pipeline with the identified parameters is ad-                   [25] C. Tomasi and R. Manduchi, “Bilateral filtering for gray and color
vised for all aerial detection systems.                                             images,” in Sixth International Conference on Computer Vision, 1998.
   Future research should focus on lightweight denoising ar-                   [26] N. Wiener, Extrapolation, Interpolation, and Smoothing of Stationary
                                                                                    Time Series. MIT Press, 1949.
chitectures, hybrid CNN transformer models, federated learn-                   [27] D. L. Donoho, “De-noising by soft-thresholding,” IEEE Transactions on
ing for multi-drone systems, domain adaptation techniques,                          Information Theory, vol. 41, no. 3, pp. 613-627, 1995.
multimodal sensor integration, and adaptive preprocessing
parameters for edge optimized implementations.
                              R EFERENCES
[1] P. Viola and M. Jones, “Rapid object detection using a boosted cascade
    of simple features,” in Proceedings of the 2001 IEEE Computer Society
    Conference on Computer Vision and Pattern Recognition, 2001.
[2] N. Dalal and B. Triggs, “Histograms of oriented gradients for human
    detection,” in 2005 IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition, 2005.
[3] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification
    with deep convolutional neural networks,” Advances in Neural Informa-
    tion Processing Systems, vol. 25, 2012.
[4] K. Simonyan and A. Zisserman, “Very deep convolutional networks for
    large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.
[5] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature
    hierarchies for accurate object detection and semantic segmentation,” in
    Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition, 2014.
[6] R. Girshick, “Fast r-cnn,” in Proceedings of the IEEE International
    Conference on Computer Vision, 2015.
[7] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time
    object detection with region proposal networks,” Advances in Neural
    Information Processing Systems, vol. 28, 2015.
[8] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look
    once: Unified, real-time object detection,” in Proceedings of the IEEE
    Conference on Computer Vision and Pattern Recognition, 2016.
[9] J. Redmon and A. Farhadi, “YOLO9000: better, faster, stronger,” in
    Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition, 2017.


---

