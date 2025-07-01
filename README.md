# 🛡️ SmartFraudX

![SmartFraudX Logo](https://img.shields.io/badge/SmartFraudX-4CAF50?style=for-the-badge&logo=datadog&logoColor=white&label=AI%20Fraud%20Detection)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Harshitraiii2005/SmartFraudX/actions)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)

---

SmartFraudX is an **end-to-end AI-powered fraud detection platform** combining real-time streaming ML, deep learning (BiLSTM), and ensemble learning for accurate transaction fraud prediction.  

🚀 **Detect suspicious activity in milliseconds.**  
🎨 **Engage users with a modern 3D animated web UI.**  
🛠️ **Train, validate, and deploy models automatically.**  
🌍 **Scalable with Docker, CI/CD pipelines, and AWS cloud storage.**

---

## 📚 Table of Contents

- [🌟 Features](#-features)
- [🚀 Demo](#-demo)
- [🖥️ Screenshots](#️-screenshots)
- [⚙️ Installation](#️-installation)
- [🔧 Usage](#-usage)
- [🧩 Technologies](#-technologies)
- [📦 CI/CD & Cloud](#-cicd--cloud)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [💬 Support](#-support)

---

## 🌟 Features

✅ **Hybrid Fraud Detection**
- **River Streaming Model** for incremental learning from live MongoDB streams.
- **BiLSTM Deep Neural Network** for sequential transaction patterns.
- **Meta-classifier Ensemble** combining predictions for higher accuracy.

✅ **Interactive 3D UI**
- Clean, animated forms.
- Real-time prediction feedback.
- Responsive design with enhanced UX.

✅ **End-to-End Pipeline**
- Data ingestion from MongoDB (simulated real-time streams).
- Validation, transformation, and model training.
- Automatic saving of trained models to AWS S3.

✅ **CI/CD & Containerization**
- **Docker** container for reproducible environments.
- **GitHub Actions workflows** for automated testing and deployment.

✅ **Scalable REST API**
- Flask-based server exposing JSON endpoints.

---

## 🚀 Demo

🎥 **Video Walkthrough:**  
[![Watch Demo](https://github.com/Harshitraiii2005/SmartFraudX/blob/main/SmartFraudDetection-GoogleChrome2025-07-0122-28-31-ezgif.com-video-to-gif-converter.gif)

---

## 🖥️ Screenshots

### Web UI

![SmartFraudX Web UI](https://github.com/Harshitraiii2005/SmartFraudX/blob/main/Smart%20Fraud%20Detection%20-%20Google%20Chrome%207_1_2025%2010_29_58%20PM.png)

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Harshitraiii2005/SmartFraudX.git
cd SmartFraudX
````

---

### 2️⃣ Create and Activate Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

---

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🔧 Usage

### 🎯 Train Models

Run the training pipeline:

```bash
python demo.py
```
```bash
python demo1.py
```
```bash
python demo2.py
```

This will:

✅ Ingest data from MongoDB or CSV
✅ Validate & transform data
✅ Train River, BiLSTM, and meta-classifier models
✅ Save models locally and to AWS S3

---

### 🏃 Start the API Server

```bash
python app.py
```

API available at:

```
http://localhost:5000
```

---

### 🌐 Access the Web Interface

```
http://localhost:5000/app
```

✅ Enter transaction details
✅ Click **Predict**
✅ View predictions & model confidence

---

## 🧩 Technologies

* **Python 3.8+**
* **Flask** — REST API backend
* **TensorFlow/Keras** — BiLSTM deep learning
* **River** — Incremental learning model
* **Scikit-learn** — Ensemble meta-classifier
* **MongoDB** — Real-time data ingestion stream
* **AWS S3** — Model storage
* **Docker** — Containerized deployment
* **GitHub Actions** — CI/CD workflows
* **HTML/CSS/JavaScript** — Web frontend
* **3D animations** — Enhanced UX

---

## 📦 CI/CD & Cloud

**Continuous Integration & Deployment**

✅ **GitHub Actions**

* On push to `main`, automated linting, testing, and Docker build.
* Can auto-deploy to AWS or other cloud providers.

✅ **Docker**

* Production-ready `Dockerfile` to run the app in an isolated container:

  ```bash
  docker build -t smartfraudx .
  docker run -p 5000:5000 smartfraudx
  ```

✅ **AWS Cloud Storage**

* Models are persisted to an S3 bucket after training.
* Example workflow:

  * Train pipeline completes
  * Models are uploaded via `boto3` to S3
  * App loads the latest model at startup

✅ **MongoDB Streaming**

* The River model consumes transaction streams in real-time from a MongoDB collection.
* Each new transaction triggers incremental learning.

---

## 🤝 Contributing

We ❤️ contributions!

1. **Fork** this repo
2. **Create** a feature branch (`git checkout -b feature/YourFeature`)
3. **Commit** your changes
4. **Push** to your branch
5. **Open** a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for full text.

---

## 💬 Support

Have questions?

* **Issues:** [GitHub Issues](https://github.com/Harshitraiii2005/SmartFraudX/issues)
* 📧 Email: [upharshi2005@gmail.com](mailto:upharshi2005@gmail.com)

---

## ⭐ Show your support

If you find this project helpful, please ⭐ star the repository and share it with others!
