---
source_pdf: 661266.pdf
pages: 10
converted_with: pdftotext -layout
---

# 661266.pdf

> Auto-converted from PDF. Prefer this file over the PDF for AI reading.

---

    IA-Med: A Machine Learning-Based System for
      Disease Prediction and Personalized Health
        Guidance with Conversational Support

     Najat El Ouahi2 , Amine Zeguendry1,2 [0000-0002-5864-5667] , Nessaiba
             Hadigui2 , Yassmin Hailala2 , and Chaimae Hanouna2
1
    Laboratory of Computer Systems and Intelligent Systems, Cadi Ayyad University,
                   UCA, Faculty of Sciences Marrakech, Morocco
      2
        LAMIGEP, EMSI, Moroccan School of Engineering Marrakech, Morocco
                              a.zeguendry@emsi.ma



       Abstract. The growing integration of Artificial Intelligence (AI) in health-
       care enables the development of intelligent diagnostic support systems.
       Our project IA-Med is a disease prediction system that, based on symp-
       toms entered by the user, provides a likely diagnosis along with personal-
       ized recommendations (treatments, precautions, nutritional advice). The
       system was developed using synthetic data to ensure privacy during the
       initial development phase. To achieve this, several machine learning mod-
       els were trained on a structured dataset. The models Random Forest
       and Support Vector Machine (SVM) were selected for their superior per-
       formance, with Random Forest achieving 99.49% accuracy on the test
       set. The system predicts possible diseases along with their probabilities
       and includes a AI Chatbot that delivers these results interactively. IA-
       Med’s novelty lies in its integrated approach combining high-accuracy
       prediction with conversational AI and personalized recommendations in
       a single platform. These results demonstrate the potential of machine
       learning to create reliable, interpretable, and user-friendly health tools,
       while highlighting the need for future validation with clinical data.

       Keywords: Disease prediction · Machine Learning · Random Forest ·
       AI Chatbot · Health recommendation system · Synthetic data


1     Introduction
Artificial Intelligence (AI) is transforming many sectors, especially healthcare,
due to the increasing complexity of medical data, the need for fast and accurate
diagnostics, and the shortage of healthcare professionals in several regions. As
medical records, imaging, and symptom data grow exponentially, AI provides
tools that can learn from large datasets and uncover hidden patterns, enabling
predictive, preventive, and personalized care [1].
    The IA-Med system addresses this pressing need by offering an accessible
health evaluation platform. Users can input their symptoms through a user-
friendly interface and receive a list of probable diseases ranked by likelihood,


---

2       N. El Ouahi et al.

along with personalized recommendations for treatment, prevention, and nutri-
tion. The inclusion of a conversational AI Chatbot enhances user engagement
and transparency, allowing even non-experts to interact with the system effec-
tively. This interactivity bridges the gap between raw AI predictions and user
understanding by providing contextual explanations and guidance.
    Novelty and Contributions: Unlike existing disease prediction systems that of-
ten focus solely on diagnostic accuracy, IA-Med introduces an integrated ecosys-
tem that combines three key innovations: (1) high-accuracy machine learning
predictions with probabilistic rankings, (2) a conversational AI assistant that
explains results and provides contextual guidance, and (3) personalized recom-
mendation engine offering tailored treatment, nutritional, and preventive ad-
vice. This holistic approach represents a significant advancement over traditional
symptom checkers by providing comprehensive support throughout the user’s
health inquiry process.
    This project also aims to tackle structural inequalities in access to healthcare,
especially in rural and underserved regions where medical infrastructure is lim-
ited. By offering immediate AI-based support, IA-Med can serve as a front-line
triage system and empower individuals with actionable insights. Furthermore,
it assists healthcare professionals by reducing the diagnostic burden, facilitating
early detection, and promoting patient education.
    Unlike rule-based expert systems that rely on static, hard-coded logic, IA-
Med adopts a data-driven approach using supervised machine learning. Algo-
rithms such as Random Forest and Support Vector Machine (SVM) are trained
on structured symptom-disease datasets, allowing the system to learn from real-
world patterns. To ensure robustness and generalizability, cross-validation tech-
niques were applied. For example, with Random Forest, the disease diagnosis is
determined by majority voting — the disease receiving the most votes from the
ensemble of decision trees is selected as the final prediction.
    IA-Med is designed with simplicity, inclusivity, and data privacy in mind. The
clean interface ensures ease of use, while ethical safeguards—such as avoiding
personal data collection and ensuring transparency—make the tool suitable for
deployment in sensitive domains like health. While IA-Med is not intended to
replace professional medical diagnosis, it serves as a powerful decision support
tool, providing fast and data-backed recommendations to guide users toward
appropriate actions.
    As AI continues to evolve, tools like IA-Med exemplify how machine learn-
ing and natural language processing can be harnessed to enhance public health
systems, promote equitable care delivery, and support evidence-based clinical
decision-making.[2][3].


2   Related Work

Recent studies have examined optimization and evaluation of machine learn-
ing models for healthcare, including hyperparameter tuning and performance
assessment, crucial for robust AI systems [5].


---

                                                 Title Suppressed Due to Excessive Length                             3

    AI has also been applied to rapid infection prediction and real-time clinical
decision support, demonstrating potential in managing health crises like COVID-
19 [2].
    Beyond healthcare, machine learning techniques support various prediction
tasks, improving understanding of supervised models and evaluation metrics [4].
    In addition to these studies, several AI-based medical chatbot systems have
been proposed in the literature. Table 1 summarizes a comparison between our
proposed IA-Med system and other recent medical chatbot solutions.


Table 1. Comparison between IA-Med and other AI-based medical chatbot systems
Project               ML Models Used Accuracy          Chatbot Integra- Medical Features             Data Type
                                                       tion
IA-Med          (Our Random       Forest, 99.49% (Ran- Gemini API (ad- Prediction + Medications + Public       synthetic
Project)              SVM                 dom Forest) vanced)           Nutrition + Exercises + Doc- dataset
                                                                        tor guidance
AI-Based     Medical Not specified        Not reported Basic interface  Prediction + General preven- Custom JSON
Chatbot for Disease                                                     tion
Prediction(2024)[10]
Med-Chatbot (2024) Decision Tree, KNN ≈80% (esti- Traditional NLP       Prediction + Basic recom- Kaggle dataset
[9]                                       mated)                        mendations
Medical Chatbot for Logistic Regression, ≈80% (vari- Standard interface Disease prediction           Multiple     public
Disease    Prediction Random      Forest, able)                                                      datasets
(2025)[8]             Decision      Tree,
                      Naive Bayes, MLP




   Several AI-based medical chatbots exist with varying accuracy, features, and
data sources. To highlight IA-Med’s advantages, we compare it with three recent
systems.
   First, it achieves higher accuracy (99.49‘%) with Random Forest, outper-
forming traditional models like Decision Tree or Logistic Regression. Second,
unlike most chatbots offering only basic guidance, IA-Med provides personalized
nutrition, exercise advice, and referrals to medical specialists. Finally, IA-Med
uses synthetic data to ensure ethical compliance and privacy, while remaining
robust and generalizable.
   This combination of accuracy, functionality, and ethical design demonstrates
IA-Med’s added value in AI-driven healthcare.


2.1       Limitations of Existing Systems

Current disease prediction tools typically suffer from several limitations: they
often provide binary diagnoses without probability estimates, lack explanatory
capabilities, offer generic rather than personalized recommendations, and oper-
ate as standalone prediction engines without integrated conversational support.
Most existing systems focus on either prediction accuracy or user interface, but
rarely integrate both aspects seamlessly.
    The IA-Med project builds on these advances by combining disease predic-
tion models, diagnostic AI Chatbot, and recommendation systems into a single
platform. Our key innovation lies in the seamless integration of these compo-
nents, creating a synergistic system where each element enhances the others.


---

4       N. El Ouahi et al.

This tool offers early disease detection and supports patient education via in-
teractive and user-friendly interfaces. Using cutting-edge AI techniques, IA-Med
aims to bridge the gap between innovative healthcare solutions and practical
applications.

3     Methodology
3.1   Dataset Description
The IA-Med system was developed using a public synthetic dataset, specifically
designed for medical AI research. These artificial data simulate real clinical cases
while guaranteeing patient confidentiality, as no personal medical information is
used. The dataset contains over 133 symptoms associated with more than 80
different diseases, representing simulated patient cases through binary vectors
indicating symptom presence (1) or absence (0). This approach preserves privacy
while enabling robust algorithm development, though future validation with real
clinical data remains necessary.

3.2   Handling Class Imbalance and Rare Diseases
Medical datasets typically exhibit significant class imbalance, with common dis-
eases being overrepresented compared to rare conditions. To ensure equitable
performance across all disease categories, IA-Med incorporates several mitigation
strategies. During training, class weighting automatically adjusts the loss func-
tion to prioritize correct classification of minority classes. The stratified train-
test split preserves the original distribution of rare diseases in both training and
evaluation phases. For prediction, a confidence thresholding mechanism requires
higher certainty levels for diseases with limited training examples, reducing po-
tential misdiagnosis of rare conditions. These approaches collectively enhance
the system’s reliability for both common and underrepresented diseases.

3.3   Data Preprocessing
Several preprocessing steps were applied to prepare the dataset for training.
First, a one-hot encoding technique was used to convert each symptom into a
binary variable, enabling the model to interpret the presence or absence of spe-
cific symptoms. Next, data cleaning was performed to remove duplicate entries
and handle any missing values that could affect model performance. Finally,
the dataset was split into training and testing sets, using an 80/20 ratio while
preserving the original class distribution to ensure a fair evaluation of the model.

3.4   Classification Models
Two supervised learning algorithms were selected for disease prediction due to
their effectiveness in handling complex classification problems. These algorithms
are capable of learning from labeled symptom data to accurately predict the
most probable diseases, making them well-suited for this medical application.


---

                                   Title Suppressed Due to Excessive Length      5

Random Forest Classifier (RFC) Random Forest is an ensemble algorithm
based on multiple decision trees. It aggregates individual predictions to obtain
a robust final decision [6].
    Formulation:

                   ŷ = MajorityVote(f1 (x), f2 (x), . . . , fT (x))

where ft (x) is the prediction of tree t, and ŷ is the final predicted class.
    Random Forest offers several advantages that make it well-suited for dis-
ease prediction tasks. It is highly accurate and robust, thanks to its ensemble
of decision trees that reduce the risk of overfitting. The algorithm can handle
high-dimensional data and non-linear relationships between symptoms and dis-
eases, making it effective even in complex medical scenarios. It is also relatively
insensitive to missing or noisy data, and provides feature importance scores,
which help interpret which symptoms contribute most to the prediction. These
strengths make Random Forest a reliable and interpretable choice for building
AI-powered diagnostic systems.


Support Vector Machine (SVM) SVM aims to determine an optimal hy-
perplane that separates classes by maximizing the margin between them. This
margin is the distance between the hyperplane and the nearest data points from
each class, called support vectors. By maximizing this margin, SVM improves
the model’s ability to generalize well to unseen data and reduce classification
errors. [7].


3.5   AI Chatbot Integration
The integrated AI Chatbot, based on the Gemini API and customized for IA-
Med, serves as a health assistant providing guidance on symptoms. It clearly
states that it is not a medical professional, does not give diagnoses, prescribe
medication, or recommend treatments, and highlights that users should consult
a qualified healthcare provider. This ethical, transparent design ensures user
safety, trust, and responsible use of AI in healthcare.


3.6   General Architecture
The IA-Med system is based on a modular architecture that integrates a machine
learning prediction engine and a conversational assistant.
    This architecture allows for smooth interaction between web components, AI
models, and data services, as illustrated below.
    As shown in Figure 1, the IA-Med system relies on a Flask-based web appli-
cation that connects the frontend interface, the ML prediction module, and the
AI assistant to medical data resources.


---

6       N. El Ouahi et al.

    Through this design, users can input symptoms, receive disease predictions,
interact with the chatbot, and obtain personalized recommendations and medical
guidance.




                 Fig. 1. General architecture of the IA-Med system



4     Results and Discussion
After training the models on the preprocessed data, their performance was eval-
uated on the test set using standard metrics: Accuracy, Precision, Recall, and
F1-score.

4.1   Classification Results
The classification results in Table 2 demonstrate that the Random Forest (RF)
model achieves perfect accuracy and precision, with only a slight decrease in
recall and F1-score. This indicates its strong ability to correctly identify disease
cases while minimizing false positives. The Support Vector Machine (SVM) also
performs well, achieving accuracy and F1-score above 94%, but is slightly less
effective than RF in this task. These results suggest that the Random Forest
model is better suited for the complex, multi-class nature of disease prediction
in our dataset.


           Model              Accuracy Precision Recall F1-score
           Random Forest (RF) 99.49%        99%        99%     99%
           SVC                 95.3%       94.7%      95.1% 94.9%
              Table 2. Performance metrics of classification models


---

                                  Title Suppressed Due to Excessive Length       7

4.2   Accuracy Analysis

The model demonstrates high reliability by correctly identifying the true disease
within the top three predicted options in over 99% of cases. For example, given
symptoms such as fever, cough, and chest pain, the system assigns the highest
probability to pneumonia (92%), followed by tuberculosis (4%), bronchitis (2%),
asthma (1%), and common cold (1%). This probabilistic ranking provides valu-
able insight into possible diagnoses, allowing healthcare professionals or users to
consider multiple potential conditions and prioritize further medical evaluation
accordingly.


4.3   AI Chatbot Efficiency

The integrated AI Chatbot demonstrated excellent response time (average 1.3
seconds). After the machine learning models provide their disease predictions,
the AI Chatbot serves as an AI health assistant to answer users’ questions about
the predicted diseases, types of doctors to consult, the reason for usage of med-
ications, and other related health inquiries. It offers informational support and
guidance, helping users better understand their situation without performing
any diagnostic or predictive functions itself.


4.4   User Interface Design and Functionality

In addition to its predictive and conversational features, IA-Med offers a user-
friendly interface with intuitive navigation, clear disease predictions, and easy
access to personalized recommendations such as nutrition, exercises, and doctor
guidance. Its combination of visual clarity and interactive elements enhances
usability, engagement, and efficient access to the AI Chatbot’s support.




                  Fig. 2. Symptom Input and Diagnosis Interface


---

8      N. El Ouahi et al.

    Figure 3 illustrates the AI chatbot that provides continuous support to pa-
tients, available 24/7




                        Fig. 3. Chatbot Assistant Interface



4.5   Novelty Assessment: Comparative Advantages


IA-Med introduces several innovative features that distinguish it from existing
disease prediction systems:
    1. Integrated Multi-Component Architecture: Unlike standalone pre-
diction tools, IA-Med seamlessly integrates machine learning prediction, con-
versational AI, and personalized recommendations into a unified platform. This
holistic approach ensures that users receive not just a diagnosis, but comprehen-
sive support including explanations, next steps, and preventive measures.
    2. Probabilistic Ranking with Confidence Scores: While most systems
provide binary outcomes, IA-Med offers probability distributions across multi-
ple potential diseases, enabling users and healthcare professionals to consider
differential diagnoses and prioritize investigations.
    3. Ethical AI Design: The system incorporates built-in safety mechanisms,
including clear disclaimers, avoidance of medication recommendations, and em-
phasis on professional consultation, addressing critical ethical concerns in med-
ical AI applications.
    4. Accessibility and Multimodal Interaction: The combination of tradi-
tional form-based input with conversational AI allows users to interact through
their preferred modality, enhancing accessibility for diverse user groups including
those with limited technical proficiency.


---

                                   Title Suppressed Due to Excessive Length        9

4.6   Discussion

The results confirm the value of using robust machine learning models with an
interactive interface. In particular, the Random Forest algorithm demonstrated
exceptional accuracy, even on multiclass datasets, thanks to its tree aggregation
mechanism, while SVM was less suited for imbalanced or complex data. Limi-
tations include reliance on synthetic data, underrepresentation of rare diseases,
and variability depending on how users describe symptoms.
    Beyond technical limitations, AI-based medical systems like IA-Med face
challenges related to legal regulations, ethics, and data privacy. In many re-
gions, diagnostic software is considered a Medical Device and must comply with
strict approval processes (e.g., FDA in the USA, CE in Europe) as well as
data protection standards like GDPR or HIPAA. IA-Med does not store per-
sonally identifiable health information; user inputs are processed in real time.
Transparency about its non-diagnostic nature is maintained, and future versions
should include robust security measures and clear consent mechanisms to ensure
regulatory compliance and user trust.
    Despite these challenges, IA-Med integrates disease prediction, medical rec-
ommendations, and an interactive AI Chatbot in a single platform, achieving
near-perfect test accuracy and providing an intuitive interface accessible even to
non-expert users.


5     Conclusion and Future Work

IA-Med represents a major step forward in AI-powered healthcare by combining
disease prediction, conversational assistance, and personalized recommendations.
Its main contribution is improving healthcare accessibility by enabling early
detection, supporting informed decisions, and making care more proactive and
patient-centered.
     Future directions include integrating RNN/LSTM models, incorporating real
clinical data, enhancing transparency through Explainable AI, and adding mul-
tilingual support for broader adoption.

References

 1. Hossain, Sorif, et al. "Machine learning approach for predicting cardiovascular
    disease in Bangladesh: evidence from a cross-sectional study in 2023." BMC car-
    diovascular disorders 24.1 (2024): 214.
 2. Shabrandi, Ali Rebwar, et al. "Fast COVID-19 Infection Prediction with In-House
    Data Using Machine Learning Classification Algorithms: A Case Study of Iran."
    Journal of AI and Data Mining 11.4 (2023): 573-585.
 3. Corfmat, Maelenn, Joé T. Martineau, and Catherine Régis. "High-reward, high-
    risk technologies? An ethical and legal account of AI development in healthcare."
    BMC medical ethics 26.1 (2025): 4.
 4. Meaney, Christopher, et al. "Comparison of methods for tuning machine learning
    model hyper-parameters: with application to predicting high-need high-cost health
    care users." BMC Medical Research Methodology 25.1 (2025): 134.
 5. Raiaan, Mohaimenul Azam Khan, et al. "A systematic review of hyperparameter
    optimization techniques in Convolutional Neural Networks." Decision Analytics
    Journal 11 (2024): 100470.


---

10      N. El Ouahi et al.

 6. Shaker, Mohammad Hossein, and Eyke Hüllermeier. "Random Forest Calibration."
    arXiv preprint arXiv:2501.16756 (2025).
 7. Nam, Nguyen Mau, Gary Sandine, and Quoc Tran-Dinh. "Lagrange Multipliers
    and Duality with Applications to Constrained Support Vector Machine." arXiv
    preprint arXiv:2501.01082 (2025).
 8. Veronica, Oltean Anisia, Ioan Daniel Pop, and Adriana Mihaela Coroiu. "Medical
    Chatbot for Disease Prediction Using Machine Learning and Symptom Analy-
    sis.".pp. 600–607. SCITEPRESS (2025).
 9. Med Chatbot Predicting Diseases From Symptoms Using NLP. Journal of Scien-
    tific and Engineering Research 7(10) (2024).
10. Zagade, Ashish, et al. "Ai-based medical chatbot for disease prediction." (2024).


---

