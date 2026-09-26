# 🩸 FingerBlood AI

### Fingerprint-Based Blood Group Classification using CNN

FingerBlood AI is a computer vision and deep learning project that analyzes fingerprint images and classifies them into eight blood-group categories using a Convolutional Neural Network (CNN).

## 🚀 Live Demo

👉 **[Open FingerBlood AI Live App](https://fingerblood-ai-3wq7zbmkpn5a7yjtypiwuh.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud** and can be accessed directly from a web browser without installing the project locally.

## 🚀 Project Overview

The application provides a Streamlit-based web interface where users can upload a fingerprint image and receive a predicted blood-group class along with prediction probabilities.

### Supported Blood Groups

- A+
- A-
- AB+
- AB-
- B+
- B-
- O+
- O-

## 📊 Model Performance

| Metric            | Value |
| ----------------- | ----: |
| Test Accuracy     | **86.62%** |
| Number of Classes | **8** |
| Input Image Size  | **128 × 128** |
| Model             | **CNN** |

## 🧠 Model Evaluation

The model was evaluated using a classification report and confusion matrix across the eight blood-group classes.

The test-set results showed an overall accuracy of approximately **87%** on 598 samples.

## 🛠️ Technology Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## 💻 Features

- 📤 Fingerprint image upload
- 🧠 CNN-based classification
- 🩸 Eight blood-group classes
- 📊 Prediction confidence
- 📈 Probability distribution
- 📋 Model evaluation and confusion matrix
- 🌐 Streamlit web interface
- 🚀 Live web deployment

## 📁 Project Structure

```text
FingerBlood-AI/
│
├── app.py
├── src/
│   └── predict.py
│
├── model/
│   ├── model.keras
│   └── confusion_matrix.png
│
├── dataset_raw/
│
├── screenshots/
│   └── fingerblood-ui.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/AryanTripathi12/FingerBlood-AI.git
cd FingerBlood-AI
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Links

- 🚀 **Live Demo:** [Open FingerBlood AI](YOUR_STREAMLIT_URL)
- 🐙 **GitHub Repository:** [FingerBlood-AI](https://github.com/AryanTripathi12/FingerBlood-AI)

## 👨‍💻 Developer

**Aryan Tripathi**

B.Tech CSE — AI & ML

GitHub:  
[https://github.com/AryanTripathi12](https://github.com/AryanTripathi12)

## ⚠️ Disclaimer

This project is developed for educational and research purposes. Fingerprint-based blood-group predictions should not be treated as medical diagnoses or as a replacement for laboratory blood typing.
