---
source_pdf: 661252.pdf
pages: 11
converted_with: pdftotext -layout
---

# 661252.pdf

> Auto-converted from PDF. Prefer this file over the PDF for AI reading.

---

     ECO-TRACK: Trash Classification and Face
       Detection Powered by Computer Vision

 Majdouline Sabri2 , Amine Zeguendry1,2[0000−0002−5864−5667] , and Rajae Ziki2
1
    Laboratory of Computer Systems and Intelligent Systems, Cadi Ayyad University
                  (UCA), Faculty of Sciences, Marrakech, Morocco
      2
        LAMIGEP, EMSI, Moroccan School of Engineering, Marrakech, Morocco
                              a.zeguendry@emsi.ma



       Abstract. Urban waste pollution often results from inadequate collec-
       tion and sorting practices. We propose ECO–TRACK, a smart waste
       management system that leverages IoT, embedded systems, and artificial
       intelligence. Smart bins equipped with ESP32 microcontrollers, ultra-
       sonic sensors, and cameras classify waste in real time using convolutional
       neural networks (CNNs) and sort it via a servo mechanism. A webcam
       can capture images of individuals littering, and face recognition is em-
       ployed to identify repeat offenders under privacy-by-design safeguards.
       Collected data are transmitted to a central server, where optimized col-
       lection routes are computed using a modified Dijkstra algorithm with the
       OpenStreetMap (OSM) API, prioritizing full bins to reduce trips, fuel,
       and emissions. The system is designed to handle diverse waste types un-
       der varied conditions and to accurately separate recyclable, compostable,
       and non-recyclable categories. Overall, ECO–TRACK improves sorting,
       monitoring, and collection efficiency in a scalable, intelligent, and envi-
       ronmentally sustainable manner.

       Keywords: Smart Waste Management · Trash Classification · CNN ·
       Face Recognition · Route Optimization


1    Introduction
ECO–TRACK merges sustainability with smart technologies to modernize urban
waste management. In many cities, population growth and consumption patterns
have increased waste production. Traditional approaches (manual sorting, static
routes) are increasingly ineffective. Improper classification leads to low recycling
rates, landfill growth, and environmental degradation.
Real world problem. Urban waste systems face pressure from rapid urbanization
and changing consumption. Manual sorting is labor intensive and error prone;
static schedules waste fuel and cause overflow or missed pickups.
Importance. Inefficiencies contribute to pollution, greenhouse gases, and public
health risks. Many smart systems rely heavily on cloud resources that can be
costly or inaccessible; there is a need for decentralized, adaptive solutions.
Gap. Prior AI/IoT solutions often depend on cloud inference, have limited clas-
sification scope, or omit end-to-end automation (physical sorting and dynamic


---

2      M. Sabri et al.

routing). Few are low-cost, real-time, and autonomous for infrastructure-limited
settings.
Problem statement. We present an edge-computing, AI-driven solution that per-
forms real-time waste classification and sorting, monitors bin status, and opti-
mizes routes dynamically to reduce inefficiency and promote recycling.
    ECO–TRACK centers on the ESP32-CAM for on device image capture and
lightweight CNN based classification of recyclable/compost/trash with low la-
tency and energy use. Servo motors actuate physical sorting, and ultrasonic
sensors track fill levels. A central server computes dynamic routes via Dijkstra
and OSM. A facial recognition module (event-triggered) can identify repeat of-
fenders to encourage accountability, implemented with data minimization and
safeguards.
Contributions. Our contributions are: (i) a fully integrated, low-cost edge system
combining on-device CNN classification (ESP32-CAM) with mechanical sort-
ing; (ii) online bin monitoring with ultrasonic sensing and dynamic OSM-based
route optimization under capacity constraints; (iii) an accountability module
with event-triggered face recognition designed with privacy-by-design; and (iv)
an empirical evaluation on two multi-class datasets with ablations and error
analysis.
Paper organization. Section 2 reviews related work. Section 3 details materials
and methods. Section 4 presents experiments and results. Section 5 concludes
and outlines future work.


2   Related Work

Automated trash classification has progressed steadily from early convolutional
neural networks (CNNs) on the TrashNet dataset, such as the work of Alwa-
teer et al. (87% accuracy), to more advanced transfer learning approaches on
lightweight architectures like MobileNet, which enabled efficient deployment on
edge devices (Rad et al., 92.3%). Other studies have emphasized both accu-
racy and explainability, for instance ResNet50 combined with Grad-CAM vi-
sualization techniques (Bai et al., 95.4%), while lightweight variants such as
EfficientNet-B0 have been adapted specifically for smart bin contexts, achieving
≈93.8% performance with lower computational demand.
    In parallel, advances in face recognition have expanded the accountability
dimension of smart waste management. Widely adopted deep models such as
FaceNet, ArcFace, YOLOFace, and InsightFace have demonstrated robust recog-
nition performance across unconstrained environments, making them suitable for
monitoring and deterrence in public spaces. Other complementary contributions,
such as VisNet for transformation-invariant object recognition and CNN-based
mask detection systems (97.05% accuracy in real time), highlight the versatility
of vision models across domains that share challenges with waste monitoring.
    Finally, in the domain of collection logistics, operations research approaches
such as Lagrangian-relaxation-based routing have shown measurable improve-
ments in reducing collection costs and travel distances. Together, these strands


---

                              ECO-TRACK: Smart Waste and Face Detection            3

of research have provided strong building blocks. However, prior works often op-
timize isolated components—classification or recognition or routing—without
offering a unified solution. ECO–TRACK addresses this gap by combining em-
bedded CNN-based classification, event-triggered face recognition, and dynamic
route optimization into a single, low-cost, and scalable framework [5,15,14,3,2,9,8,6,1,4,10].
A concise comparison of representative studies is provided in Table 1.



                  Table 1: Comparison of related studies (abridged).
Study                   Approach / Model        Performance      Key Contribution
Alwateer et al. [13]    CNN on TrashNet         87% acc.         Early CNN; class im-
                                                                 balance mitigation
Rad et al. [13]         MobileNet TL            92.3% acc.       Edge-optimized trans-
                                                                 fer learning
Thung & Yang [11]       CNN + image proc.       88.7% F1         Hybrid features for
                                                                 classification
Bai et al. [3]     ResNet50 + Aug + Grad- 95.4% acc.             Accuracy + inter-
                   CAM                                           pretability
Deep FR models [4] FaceNet/YOLOFace       High (varies)          Littering accountabil-
                                                                 ity



3     Materials and Methods
We outline dataset preparation, model architecture/training, system integration,
route optimization, and model selection.

3.1   Dataset Preparation
Two Roboflow datasets were used: the first [7] with recyclable and trash; the
second [12] adds compost. Images span varied environments, lighting, occlusions,
and orientations. Each image includes YOLO/COCO format bounding boxes.
This diversity supports robust generalization. A compact summary of Dataset 1
is given in Table 2.



                       Table 2: Dataset 1 (Roboflow) summary.
Property                      Details
#Images                       3,200
Classes                       Recyclable; Trash
Split (Train/Val/Test)        70 / 15 / 15
Distribution (%)              Recyclable 45%; Trash 55%


Figure 1 presents sample images from the dataset used in this study, illustrating
the three primary waste classification categories: recyclable, trash, and compost.


---

4        M. Sabri et al.

These examples highlight the diversity and complexity of visual features within
each class, including variations in lighting, occlusion, background clutter, and
object deformations.




    Fig. 1: Representative images across classes: recyclable, trash, and compost.



3.2     Model Architecture and Training

We used YOLOv8 for high-performance detection (anchor-free with enhanced
backbone and auto-label assignment). Preprocessing includes resizing to 640×640,
normalization, and augmentations such as flip, scale, brightness, and rotation.
Transfer learning is applied from COCO weights, followed by fine-tuning on our
datasets with class-specific bounding boxes. Mean Average Precision (mAP) is
reported at IoU thresholds (mAP@0.5 and mAP@0.5:0.95), where IoU is Inter-
section over Union. The main components are illustrated in Figure 2 for trash
classification and in Figure 3 for face detection.




              Fig. 2: Trash classification model architecture overview.


---

                                                            ECO-TRACK: Smart Waste and Face Detection                                                                               5

The waste classification module (Figure 2) is complemented by a face detec-
tion component, shown next, which enables accountability and user monitoring
within the system.




                                                       Fig. 3: Face detection model.


3.3   System Integration: Hardware–Software Pipeline

The end-to-end hardware/software flow of ECO–TRACK is shown in Figure 4.
The ESP32-CAM captures an item, runs a compact CNN locally (quantized for
TensorFlow Lite for Microcontrollers, TFLM), and triggers the servo to sort.
Ultrasonic sensors publish fill-level events; when thresholds are exceeded, bin
states are sent to the server. The server aggregates states, queries OSM, and
computes dynamic routes (Section 3.4). Accountability images, when generated,
are encrypted, access-controlled, and retained briefly (see Ethics).


       Edge / Bin                                                                                                                                  Backend / City Ops


                                              frames                                        class label                                                   Secure storage
                  ESP32 CAM                                      Lightweight CNN                             Servo mechanism                       (Encrypted, access controlled)
                (Image capture)                             (TFLM, on device inference)                      (Physical sorting)


                                                                                                                                                              Central server
                                                                                                              encrypted upload                             (State aggregation)
                                            level/trigger       littering event               events                                       bin states
               Ultrasonic sensors                                    Bin state                              Gateway / Network
               (Fill level sensing)                             (threshold & event)                          (Wi Fi/LTE/MQTT)
                                                                                                                   route plan
                                                                                                              Driver app / Truck                             Routing engine
                                                                                                          (Route plan, turn by turn)                    (Modified Dijkstra + OSM)

                                                                  Event triggered
                                                                   face capture

      Notes: TFLM = TensorFlow Lite for Microcontrollers; OSM = OpenStreetMap. Event triggered capture; short retention; access logging.




                       Fig. 4: Hardware–software pipeline of ECO–TRACK.


---

6       M. Sabri et al.

3.4   Route Optimization

Let G = (V, E) be the OSM road graph with      Pnonnegative travel costs cuv . We
compute a depot-rooted route minimizing           cuvPsubject to: (i) visiting bins
with fill-level ≥ τ , (ii) truck capacity constraint i wi ≤ C (volume estimate
wi , capacity C), and (iii) return to depot. We use a modified Dijkstra to obtain
a feasible near optimal sequence by capacity-aware pruning and edge reweight-
ing with bin priority (fill-level and organic content). This heuristic yields large
fuel/time gains and supports online updates [10].


3.5   Model Selection: YOLOv8 vs ResNet

To motivate the deployment choice, Table 3 contrasts YOLOv8 and ResNet
across dimensions such as primary task, output type, computational overhead,
and feasibility on edge devices. While YOLOv8 excels in object detection with
high accuracy, its resource demands make it less suitable for low-power mi-
crocontrollers. In contrast, ResNet offers a classification-focused backbone with
modular depth and established transfer learning benefits, which can be adapted
more efficiently for constrained hardware. We therefore derive a lightweight CNN
inspired by ResNet principles, incorporating model quantization and pruning to
reduce memory footprint and energy cost, and specifically tailoring deployment
for ESP32-CAM, ensuring real-time inference under stringent edge constraints.



        Table 3: Simplified comparison between YOLOv8 and ResNet.
Feature                            YOLOv8                     ResNet
Primary Task                       Object Detection           Image Classification
Output                             Boxes + labels             Single class label
Speed                              Real-time (with GPU)       Task-dependent
Architecture                       Single-stage detector      Deep CNN (residual)
Performance                        High mAP                   High accuracy
Hardware Needs                     High (GPU/TPU)             Moderate–High
Edge Feasibility                   Low on MCU                 Moderate (with compact
                                                              CNN)




4     Experiments and Results

We fine-tuned two YOLOv8 models on distinct datasets (two-class and three-
class). We report precision, recall, mAP, F1, PR curves, and confusion matrices,
and summarize validation performance.


---

                            ECO-TRACK: Smart Waste and Face Detection            7

4.1   Training Settings and Metrics
Precision P = T PT+F
                   P                    TP                         PR
                      P , Recall R = T P +F N , and F1 score = 2· P +R are the main
evaluation metrics used. Additionally, we report mean Average Precision (mAP)
at IoU thresholds of 0.5 and the range 0.5:0.95. The training hyperparameters
are listed in Table 4.



         Table 4: Training hyperparameters (classification/detection).
                 Parameter      Value
                 Learning Rate 0.001
                 Optimizer      Adam
                 Batch Size     32
                 Epochs         50
                 Loss           Categorical Cross-Entropy
                 Activation     ReLU (hidden), Softmax (output)
                 Early Stopping Patience 5 (val. loss)




4.2   Performance
Figure 5 provides a detailed view of the validation performance through Preci-
sion–Recall (PR) curves for each class as well as the F1–confidence curve, which
is particularly useful for identifying an operating threshold that balances pre-
cision and recall in practical deployment. These curves highlight how well the
model is able to maintain high recall without sacrificing too much precision and
vice versa, offering a deeper understanding of trade-offs across different thresh-
olds. In addition, Figure 6 illustrates the confusion matrix for the three-class
problem, revealing the dominant sources of misclassification, such as the overlap
between compost and trash due to their similar visual features. Finally, Table 5
summarizes the quantitative results by reporting precision, recall, and mean Av-
erage Precision (mAP) scores at different IoU thresholds, thereby providing a
comprehensive numerical benchmark of the model’s effectiveness across all cat-
egories.
    Together, these results establish both the strengths of the system in han-
dling recyclable items and the challenges that remain in reliably distinguishing
between compost and trash in real-world conditions. They also demonstrate
that while the model performs competitively relative to prior work, further re-
finements are required to close the gap with state-of-the-art accuracy levels.
These insights are crucial for guiding future improvements, such as integrating
multi-modal sensing (e.g., weight, gas, or RFID-based inputs) to complement
vision-based recognition and reduce ambiguity in visually similar waste types.
Moreover, they emphasize the value of advanced augmentation strategies and
domain adaptation techniques to enhance robustness under varied lighting con-
ditions, occlusions, and environmental noise. By analyzing the interplay between


---

8      M. Sabri et al.

qualitative error patterns and quantitative performance metrics, these evalua-
tions not only validate the feasibility of ECO–TRACK in constrained edge set-
tings but also highlight concrete directions for achieving more reliable, scalable,
and environmentally impactful deployments.




      (a) PR curves (validation).                  (b) F1–confidence curve.

             Fig. 5: Validation PR curves and F1–confidence curve.




               Fig. 6: Confusion matrix for the three-class model.


---

                            ECO-TRACK: Smart Waste and Face Detection            9

    Table 5 reports validation results, with the recycle class showing the best
precision (0.762) and recall (0.806). Compost and trash achieve lower scores due
to visual similarity, though recall remains solid across all classes, ensuring most
items are detected despite occasional mislabels.



             Table 5: Validation metrics (Dataset 2, three classes).
                  Class Precision Recall mAP@0.5 mAP@0.5:0.95
                Compost 0.593 0.705       0.699     0.652
                Recycle 0.762 0.806       0.840     0.723
                 Trash   0.576 0.722      0.698     0.660
                 Overall 0.644 0.744      0.746     0.678



Interpretation and Discussion. The proposed YOLOv8 model achieves an over-
all mAP@0.5 of 74.6%, which demonstrates strong performance given the con-
straints of edge deployment on low-cost hardware. Among the three categories,
the recycle class performs the best, reaching a precision of 76.2% and a recall of
80.6%, indicating that recyclable materials are both consistently detected and
rarely missed. In contrast, the compost and trash classes exhibit comparatively
lower precision, largely due to their overlapping visual characteristics—such as
similar textures, colors, and degrees of decomposition—that make them harder
to distinguish reliably. Despite this, recall values remain higher than precision
across all categories, meaning the model favors completeness of detection (min-
imizing false negatives) at the expense of occasional false positives, which is
generally preferable for waste-sorting scenarios where missed detections can un-
dermine sustainability goals.

Ethical and Privacy Considerations
The implementation of facial recognition raises important ethical questions.
ECO–TRACK follows privacy-by-design and data minimization: (i) event-triggered
capture (e.g., littering) rather than continuous recording; (ii) short retention win-
dows with deletion of non-violating footage; (iii) encrypted storage and transport
with access restriction and audit logs; and (iv) visible signage/notice where appli-
cable. Only images relevant to repeated violations are retained, and identities are
not shared beyond authorized municipal stakeholders. Deployments should align
with applicable data protection regulations (e.g., GDPR-like principles), includ-
ing lawful-basis assessment and Data Protection Impact Assessment (DPIA).
Misidentification and bias risks are mitigated through calibrated thresholds, pe-
riodic human-in-the-loop review, and fairness audits on diverse datasets.

Limitations
The evaluation is image-centric; performance can degrade under extreme light-
ing or heavy occlusion. Compost vs. trash confusion persists. ESP32-CAM con-


---

10      M. Sabri et al.

straints limit model depth and input resolution. Future work will examine sensor
fusion, adaptive thresholds, and cross-city generalization.


Reproducibility

Training configurations and model checkpoints can be provided to enable exact
reruns of reported experiments.


5    Conclusion and Future Work

ECO–TRACK combines embedded sensors, computer vision, and AI to auto-
mate sorting and optimize collection. A compact CNN on ESP32-CAM performs
real-time classification (recyclable/compost/trash), ultrasonic sensors track fill
levels, and a modified Dijkstra algorithm computes efficient routes over OSM,
reducing fuel use and emissions. An event-triggered face-recognition module sup-
ports accountability under explicit privacy safeguards. Results show reliable per-
formance for real-world deployment.
    Future work includes fully autonomous smart bins and robotic collection
vehicles, larger and more diverse datasets, city-scale analytics, and improved edge
performance. We also plan to explore federated learning for privacy-preserving
updates across distributed bins.


References
 1. Abadi, M., et al.: Tensorflow: Large-scale machine learning on heterogeneous sys-
    tems (2015), software available from tensorflow.org
 2. Cowger, W., et al.: Trash ai: A web gui for serverless computer vision analysis of
    images of trash. Journal of Open Source Software 8(89), 5136 (September 2023)
 3. Dong, Z.: Intelligent garbage classification system based on computer vision. In-
    ternational Core Journal of Engineering 7(4) (April 2021)
 4. K. He, X. Zhang, S.R., Sun, J.: Deep residual learning for image recognition. Pro-
    ceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR) pp. 770–778 (2016)
 5. Keisler, J.: Automated deep learning: Algorithms and software for energy sustain-
    ability (2025)
 6. Redmon, J., Farhadi, A.: Yolov3: An incremental improvement. arXiv preprint
    arXiv:1804.02767 (2018), accessed: 2025-05-16
 7. Roboflow: trash-sorter-all-classes, accessed: 2025-05-16
 8. roboflow: Trash sorter all classes dataset, accessed: 2025-05-16
 9. Roboflow: Vehicle routing with time windows: Two optimization algorithms, ac-
    cessed: 2025-05-16
10. Roboflow: waste-detection-dpmnj, accessed: 2025-05-16
11. S. Ren, K. He, R.G., Sun, J.: Faster r-cnn: Towards real-time object detection with
    region proposal networks. IEEE Transactions on Pattern Analysis and Machine
    Intelligence 39(6), 1137–1149 (2017)
12. Ultralytics: Yolov8, accessed: 2025-05-16


---

                             ECO-TRACK: Smart Waste and Face Detection               11

13. Viola, P., Jones, M.: Rapid object detection using a boosted cascade of simple
    features (2001)
14. Yang, M., Thung, G.: Classification of trash for recyclability status. CS229 Project
    Report (2016), 2016.1:3
15. Yu, Y.: A computer vision based detection system for trash bins identification
    during trash classification. Journal of Physics: Conference Series 1617(1) (2020)


---

