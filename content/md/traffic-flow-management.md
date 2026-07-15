---
source_pdf: 661256.pdf
pages: 8
converted_with: pdftotext -layout
---

# 661256.pdf

> Auto-converted from PDF. Prefer this file over the PDF for AI reading.

---

     AI-Driven Traffic Flow Management System:
      Real-Time Blockage Detection, Emergency
     Prioritization, and Road Hazard Monitoring

Youssef Aitrais2 , Amine Zeguendry1,2 , Ayyoub Raji2 , Elmehdi El Ksakes2 , and
                             Yassine ElOuarnaz2
1
    Laboratory of Computer Systems and Intelligent Systems, Cadi Ayyad University,
                   UCA, Faculty of Sciences Marrakech, Morocco
      2
        LAMIGEP, EMSI, Moroccan School of Engineering Marrakech, Morocco
                              a.zeguendry@emsi.ma



       Abstract. This study introduces an AI-driven Traffic Flow Manage-
       ment System designed to enhance urban mobility and safety by address-
       ing critical issues like congestion, vehicle-induced blockages, and road
       infrastructure defects. The solution integrates real-time vehicle count-
       ing, blockage detection based on the traffic flow rate equation, emer-
       gency vehicle prioritization, and crowdsourced pothole detection. A core
       feature is the system’s ability to dynamically adjust the duration of IoT-
       controlled traffic signals in response to detected blockages, optimizing
       traffic throughput.
       The system employs computer vision models for detection tasks. The
       Emergency Vehicle Detection model achieves 94.5% precision, 92.9% re-
       call, and 96.1% mAP@50. Pothole detection attains 80.8% precision,
       87.2% recall, and 88.5% mAP@50. The dynamic signal control algorithm
       proved effective in managing simulated traffic anomalies. The architec-
       ture is scalable, supporting real-time processing and sub-5-minute emer-
       gency response capabilities.
       The findings demonstrate the viability of a data-centric, integrated sys-
       tem for achieving proactive traffic control, offering a scalable blueprint
       for managing dynamic urban traffic and improving both safety and effi-
       ciency.

       Keywords: Traffic Flow · Blockage Detection · Dynamic Signal Control
       · AI Frameworks · YOLO · Computer Vision · IoT · Adaptive Traffic
       Control · Emergency Preemption


1     Introduction

Urban traffic congestion and sudden blockages severely impact mobility and
safety. Traditional static traffic control systems cannot respond dynamically to
critical events like road blockages or the immediate need for emergency vehicle
passage. This creates a clear requirement for intelligent, real-time traffic man-
agement.


---

2       Y. Aitrais et al.

     Recent technological convergence, specifically Artificial Intelligence (AI) for
computer vision and Internet of Things (IoT) for infrastructure control, offers
the opportunity to create truly adaptive urban systems.
     This study proposes a novel AI-Driven Traffic Flow Management System.
The core innovation lies in integrating: 1. Vehicle Counting and Flow Analy-
sis to detect blockages using the traffic flow rate equation. 2. Dynamic Signal
Control to instantly adjust IoT traffic light durations upon blockage detection.
3. Emergency Prioritization via automated signal switching. 4. Road Hazard
Detection (potholes) for infrastructure maintenance.
     The remainder of this paper details the system’s architecture, methodology,
performance validation, and contribution to enhancing urban safety and mobil-
ity.


2   Related Work

The Highway Safety Manual (HSM) [1] is a leading reference for accident predic-
tion, by presenting Safety Performance Functions (SPFs) and Crash Modification
Factors (CMFs). International projects such as RIPCORD-iSEREST and RIS-
MET have validated and adapted HSM methods, employing various statistical
models and emphasizing the importance of calibration.
     Recent advances in artificial intelligence have revolutionized traffic manage-
ment and road safety applications. Vision-based intelligent traffic light man-
agement systems using deep learning models like Faster R-CNN have shown
promising results in real-time traffic optimization [2]. Work leveraging YOLOv8
on mobile platforms has demonstrated the high accuracy (93%) and feasibility
of real-time urban traffic data collection and vehicle counting, establishing the
foundation for our flow analysis approach [3]. Furthermore, advanced studies
analyze complex factors like gender differences in casualty trends to inform tar-
geted road safety policies and interventions [4]. YOLO-based approaches have
been extensively applied for traffic sign detection [5] and pothole detection, with
recent studies achieving 93.0% precision, 91.6% recall, and 96.3% mAP using
built-in vehicle technologies [6]. Tiny-YOLOv4 has demonstrated effective real-
time pothole detection, achieving 90% accuracy and processing at 31.76 FPS
[7].
     Crowdsourcing-based multi-sensor fusion approaches have been proposed for
large-scale road pothole detection [8], while real-time traffic incident detection
systems demonstrate immediate response capabilities for preventing secondary
accidents [9]. The convergence of AI technologies with traditional traffic man-
agement approaches represents a significant advancement in road safety sys-
tems. While current literature demonstrates considerable progress in isolated
traffic management system components, a key challenge persists: the absence of
comprehensive solutions that seamlessly combine diverse AI technologies with
crowdsourced data for truly holistic road safety management. This paper intro-
duces a novel system designed to address this deficit, integrating real-time object
detection, crowdsourced reporting, and adaptive traffic control into a cohesive


---

                                         AI-Driven Traffic Flow Management          3

framework. International road safety initiatives provide a useful framework for
comparison, summarized in Table 1 below.


        Table 1. Summary of Road Safety Strategies in Selected Jurisdictions

      Jurisdiction      Key Strategies               Outcomes
      Western Australia Safe System Matrix, in- Predicted 50% reduction in
                        frastructure upgrades, speed casualties
                        control
      Netherlands       Ex-ante/ex-post evaluation, Identified gaps in injury re-
                        target adjustments           ductions
      Sweden            Vision Zero, 2+1 roads, Crash reductions of 30–44%
                        speed cameras
      Switzerland       Via Sicura, cost-benefit 33% fatality reduction tar-
                        analysis                     get




3     Materials and Methods

The system adopts a hybrid approach, combining AI-driven computer vision for
detection and a mathematical model for blockage inference. Video streams from
cameras are processed using state-of-the-art object detection models (YOLOv11),
trained on datasets sourced from Roboflow [10], including "Smart Trafic 2" [11]
for vehicle classes and the "Pothole Detection Project" [12] for road defects.


3.1    Implementation Overview and Detection Logic

The system’s core functionality revolves around real-time object detection and
dynamic decision-making.
    The entire system architecture is structured as a processing pipeline, as de-
tailed in Figure 1.




                   Fig. 1. Counting and Detection System Pipeline


---

4       Y. Aitrais et al.

    The overarching framework, depicted in Figure 1, utilizes the YOLO (You
Only Look Once) algorithm for high-speed, accurate detection. The detected
object data is stored in MongoDB for dynamic visualization and analysis, facil-
itating the blockage logic, emergency prioritization, and hazard reporting.


Vehicle Counting and Flow Rate Analysis for Blockage Detection The
video analysis pipeline first performs real-time vehicle counting. The detected
vehicle data (class, location, timestamp) is used to calculate the localized traffic
flow rate (q) and density (k) within a monitored segment.
    A blockage is inferred when the observed flow rate (qobs ) significantly drops,
or the density (kobs ) sharply rises, while the segment’s occupancy time exceeds
a predefined threshold. The system uses a simplified model, monitoring vehicle
throughput per lane over set intervals:

                                          N
                                    q=
                                         ∆t · L
    where N is the number of vehicles passing a detection line, ∆t is the mea-
surement interval, and L is the number of lanes. When qobs falls below an alert
threshold qcrit (indicating congestion or a standstill), an alert is generated, and
the system initiates its dynamic signal control protocol. Upon blockage alert, the
system sends a signal to the IoT-controlled traffic light to change the duration
of the green phase for the affected or bottlenecked direction, actively mitigating
the congestion’s spread and managing the blockage. Figure 2 illustrates a typical
vehicle that could cause a blockage on a narrow road segment.


Emergency Vehicle Prioritization The system continuously monitors for
emergency vehicles (specifically ambulances, fire trucks, and police vehicles).
This function is critical for public safety and is independent of the general flow
rate analysis. When an emergency vehicle is detected, a signal is immediately
sent to the IoT traffic signal to change the state of the LED to green for the
vehicle’s path, overriding standard timing for rapid transit. This preemption re-
lies on high-speed computer vision (YOLO) models to ensure detection within
milliseconds. An encrypted, low-latency signal is then transmitted to the traffic
controller, which executes a safety sequence to clear the vehicle’s path, dramati-
cally reducing response times. The system prioritizes immediate green signal ex-
tension and coordinated clearance of conflicting approaches to ensure the safest
passage. A critical safety measure involves checking adjacent intersection sta-
tuses to initiate a cascade of green lights, effectively creating a clear emergency
corridor. The traffic signal then maintains the green phase until the vehicle has
completely traversed the detection zone, at which point the system logs the
successful preemption event, duration, and route. Following the passage of the
emergency vehicle, the adaptive control system immediately reverts to normal
flow management or initiates a short transition phase to quickly and safely return
the intersection to optimized traffic control, minimizing residual congestion.


---

                                        AI-Driven Traffic Flow Management        5

    Detection of an emergency vehicle is confirmed visually, as shown in Figure
2 for a fire truck.




          Fig. 2. YOLO Detection of an emergency vehicle (a fire truck).




Pothole Detection and Alert System The computer vision system also per-
forms Pothole Detection for road hazard monitoring. Upon detection of road
defects, the system generates an immediate alert within the system to relevant
personnel, detailing the location and severity for prompt infrastructure mainte-
nance scheduling. The system’s ability to visually confirm road defects is critical
for maintenance. An example of pothole detection is illustrated in Figure 3.




                    Fig. 3. Detection of road defects (potholes).


---

6       Y. Aitrais et al.

4     Results and Discussion

The developed system demonstrates state-of-the-art performance in real-time
flow analysis and hazard identification, showcasing its efficiency in a dynamic
urban context.
    Dynamic Signal Efficacy: The integration of vehicle counting and the blockage
logic proved highly effective. In simulated urban scenarios modeling traffic flow
anomalies, the dynamic signal control protocol—which changes the duration of
the IoT device signal upon blockage detection—indicated a significant gain in
traffic throughput and congestion resolution by actively managing the green light
timer to clear the bottleneck.
    Emergency Prioritization Response: The system’s ability to immediately de-
tect an emergency vehicle and send a signal to change the IoT LED to green for
priority passage was validated with a strong mAP@50 of 0.961, confirming its
reliability for life-critical prioritization.
    Hybrid Data Fusion: Crowdsourced pothole reports and real-time AI detec-
tions are cross-validated, stored in MongoDB with sub-200ms query latency. This
hybrid approach contributes to an overall 92.4% validation accuracy for hazard
and incident reporting. The immediate alert mechanism upon pothole detection
ensures timely maintenance response.
    Scalable Architecture: The system is designed to handle high data through-
put, capable of processing vehicle counts and flow rates from over 50 concurrent
video feeds at a smooth 60 frames per second (fps). It maintains a design goal of a
sub-5-minute end-to-end response capability for emergency vehicle prioritization
and blockage alerts.


4.1   Performance Metrics

The trained models demonstrated strong performance across key metrics, sum-
marized in Table 2.


                    Table 2. Model Training Performance Metrics

             Model                         Precision Recall mAP@50
             Emergency Vehicle Detection     0.945    0.929    0.961
             Pothole Detection               0.808    0.872    0.885



    The Emergency Vehicle Detection model achieved strong results, demonstrat-
ing a precision of 0.945, recall of 0.929, and an mAP@50 of 0.961, confirming its
reliability for life-critical prioritization. While practical, the Pothole Detection
model achieved slightly lower scores (Precision: 0.808, mAP@50: 0.885), indicat-
ing potential for further refinement in diverse lighting and road conditions.


---

                                           AI-Driven Traffic Flow Management            7

5    Conclusion
This paper presents an integrated Traffic Flow Management System that syn-
ergizes AI-driven flow analysis, dynamic signal control, and emergency prioriti-
zation. The system successfully leverages vehicle counting and the traffic flow
rate equation to achieve real-time blockage detection, triggering immediate sig-
nal duration adjustments via connected IoT devices to actively manage the flow.
Furthermore, it incorporates high-accuracy emergency vehicle preemption, which
sends a signal to change the IoT traffic light to green upon detection. This dy-
namic control capability, combined with the hazard alert system for potholes,
proved effective in addressing simulated traffic congestion.
    Future work will focus on integrating predictive flow modeling using spatio-
temporal deep learning architectures to anticipate blockages before they fully
materialize. Further efforts will be dedicated to deploying the system on edge
computing platforms for ultra-low latency operation and developing comprehen-
sive policy integration APIs for seamless adoption by municipal traffic agencies.

Acknowledgements The authors thank the Ecole Marocaine des Sciences de
l’Ingénieur for supporting this research and providing access to computational
resources.


References
 1. AASHTO: Highway Safety Manual. American Association of State Highway and
    Transportation Officials (2010)
 2. Abbas, S., et al.: Vision based intelligent traffic light management system using
    Faster R-CNN. CAAI Transactions on Intelligence Technology (2024)
 3. Mobile Application Utilizing YOLOv8 for Real-Time Urban Traffic Data Collec-
    tion. E3S Web of Conferences 601, 00077 (2025)
 4. Bačkalić, S., et al.: Analysis of casualties of young car drivers in traffic accidents
    by gender differences. Road Safety Research (2025)
 5. YOLO-BS: A traffic sign detection algorithm based on YOLOv8. Scientific Reports
    (2024)
 6. Ruseruka, et al.: Augmenting roadway safety with machine learning and deep
    learning: Pothole detection and dimension estimation using in-vehicle technolo-
    gies. Transportation Engineering (2024)
 7. Asad, et al.: Pothole Detection Using Deep Learning: A Real-Time and AI-on-the-
    Edge Perspective. Advances in Civil Engineering (2022)
 8. Sustainable Road Pothole Detection: A Crowdsourcing Based Multi-Sensors Fusion
    Approach. Sustainability (2023)
 9. A Real-Time Computer Vision Based Approach to Detection and Classification of
    Traffic Incidents. Big Data and Cognitive Computing (2023)
10. Roboflow: Computer vision platform. https://roboflow.com (2023)
11. final-project-mfmvs:       Smart        Trafic      2.      Roboflow         Universe.
    https://universe.roboflow.com/final-project-mfmvs/smart-trafic-2 (2024)
12. pothole-detection-project-owbtl:           Pothole        Detection           Project.
    https://universe.roboflow.com/20602-tsfgq/pothole-detection-project-owbtl
    (2024)


---

8       Y. Aitrais et al.

13. Elvik, R., Hoye, A., Vaa, T., Sorensen, M.: The Handbook of Road Safety Measures.
    2nd edn. Emerald Group Publishing, Bingley (2009)
14. Hauer, E.: On prediction in road safety. Safety Science 48(9), 1111–1122 (2010)
15. Weijermars, W., van Berkum, E.: Road safety forecasting and ex-ante evaluation
    of policy in the Netherlands. Transportation Research Part A 52, 64–72 (2013)
16. Siegrist, S.: Forecasting the effectiveness of national road safety programmes. Safety
    Science 48(9), 1106–1110 (2010)
17. Corben, B., Logan, D., Fanciulli, L., Farley, R., Cameron, I.: Strengthening road
    safety strategy development ’Towards Zero’ 2008–2020. Safety Science 48(9), 1085–
    1097 (2010)
18. Belin, M.A., Tillgren, P., Vedung, E.: Vision Zero – a road safety policy innova-
    tion. International Journal of Injury Control and Safety Promotion 19(2), 172–179
    (2011)


---

