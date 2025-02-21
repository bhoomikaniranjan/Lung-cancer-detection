# 🚀 PulmoTrainer: AI-Powered Lung Cancer Detection

## 🌟 Project Overview
Lung cancer is one of the deadliest diseases, and early detection can significantly improve survival rates. This AI-driven Lung Cancer Detection system leverages Deep Learning to analyze CT scan images and predict cancer presence efficiently. The project integrates a **3D Convolutional Neural Network (CNN)** and a user-friendly **Graphical User Interface (GUI)** for seamless operation.

Using a dataset of thousands of high-resolution lung scans, this model accurately determines when lesions in the lungs are cancerous. This helps in reducing false positives, providing early access to life-saving interventions, and giving radiologists more time to focus on their patients.

## 🏆 Key Features
- 📂 **Easy Data Import:** Load and process DICOM CT scan images effortlessly.
- 🔄 **Smart Preprocessing:** Automated resizing, normalization, and feature extraction.
- 🤖 **Advanced CNN Model:** A powerful 3D CNN architecture tailored for medical imaging.
- 📊 **Real-Time Training Metrics:** Monitor training progress with accuracy and loss graphs.
- 🎨 **Interactive GUI:** Simple yet effective interface for non-technical users.

## 🏥 Dataset Information
We have taken **50 patients** as a sample dataset for training and validation.

🔗 **Sample Dataset Images:** [Click Here](https://qnm8.sharepoint.com/:f:/g/Ep5GUq573mVHnE3PJavB738Bevue4plkiXyNkYfxHI-a-A?e=UVMWne)

### Workflow
1. **Import Data:** Click "Import Data" to load CT scans.
2. **Preprocess Data:** Click "Pre-Process Data" to prepare images.
3. **Train Model:** Click "Train Data" to start CNN model training.
4. **Predictions:** The trained model detects cancerous and non-cancerous cases.

## 💡 Model Highlights
- 📌 **3D CNN layers** designed for volumetric image analysis.
- 📌 Input shape: `(10, 10, 5, 1)` (resized slices of CT scans).
- 📌 Uses **Adam Optimizer** and **Categorical Cross-Entropy Loss**.
- 📌 Predicts **Cancerous (1) or Non-Cancerous (0)** cases.

## 📈 Expected Outcomes
- Efficient lung cancer detection using AI.
- Reduced diagnostic time with automated predictions.
- Enhanced medical imaging analysis with deep learning.

## 🎥 Demo Video
[Watch Demo](https://user-images.githubusercontent.com/68781375/162584302-e0a58cfe-9a1d-45a1-816e-6bfadf45821a.mp4)

## 📸 Output Screenshots
![OutputScreenshot-1](https://user-images.githubusercontent.com/68781375/162584315-359fba81-6827-437f-ab54-b8dee534f1d8.JPG)

Let's revolutionize lung cancer detection with AI! 💙🩺
