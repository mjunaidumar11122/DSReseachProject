
# Credit Card Fraud Detection Project

##  Overview

This project focuses on detecting fraudulent credit card transactions using machine learning and deep learning models. Three models were implemented and compared:  
- **Stochastic Gradient Descent (SGDClassifier)**  
- **K-Nearest Neighbors (KNN)**  
- **Long Short-Term Memory (LSTM)**

The objective is to identify which model offers the best balance between accuracy, precision, recall, and computational efficiency for real-world fraud detection applications.


## Project Structure

- `CreditCardFraud.ipynb` — Main Google Colab notebook containing all code, analysis, and model evaluations.
- `KNN.png`, `LSTM.png`, `SGD.png` — Confusion matrix plots for each model.
- `README.md` — Project description and instructions (this file).


## Dataset

- **Source:** [Hugging Face](https://huggingface.co/datasets/tanzuhuggingface/creditcardfraudtraining)
- **Description:** Synthetic yet realistic credit card transaction data with balanced classes (fraudulent and non-fraudulent).
- **Features:** Transaction time, amount, card number (anonymized), latitude, longitude, and fraud label.

---

## Models Used

| Model              | Description |
|--------------------|-------------|
| **SGDClassifier**   | Fast linear model suitable for large-scale datasets. |
| **K-Nearest Neighbors** | Instance-based method capturing local data structure. |
| **LSTM Neural Network** | Deep learning model that captures sequential patterns and long-term dependencies in transaction behavior. |

---

## Methodology

- Preprocessing included feature scaling and dataset balancing.
- Data split into **training** and **testing** sets.
- Models were trained, validated, and evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **Confusion Matrices**

**Conclusion:**  
The LSTM model delivered the best performance, successfully detecting fraudulent activities with high accuracy and minimal false positives and negatives.


##  Technologies

- Python 3.11
- Google Colab
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn
- Hugging Face Datasets

---

## Ethical Considerations

The dataset is fully anonymized and publicly available, ensuring compliance with GDPR and privacy standards. No personal identifiable information (PII) was used during the project.


## Future Work

- Implementation of ensemble models (e.g., Random Forest, XGBoost).
- Exploration of real-time fraud detection systems.
- Integration of feature engineering and hyperparameter optimization for further performance enhancement.


##  Acknowledgments

Special thanks to Hugging Face for providing open-access datasets and to the research community for continued advancements in fraud detection methods.

