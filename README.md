# Target-Oriented Multi-Aspect Based Sentiment Analysis (ABSA) for Laptops

A complete end-to-end NLP pipeline and interactive dashboard for extracting and analyzing aspects, sentiments, and causes from laptop reviews. 

This project uses a combination of fine-tuned Transformer models to not only predict if a review is positive or negative, but to understand *what specific aspect* (e.g., battery life, screen, keyboard) the user is talking about, and *why* they feel that way.

## ✨ Features
- **Aspect Extraction (BERT):** Fine-tuned `bert-base-uncased` to detect laptop-specific terms (battery, screen, touchpad, price, etc.) within raw text.
- **Sentiment Classification (RoBERTa):** Fine-tuned `roberta-base` to classify the sentiment (Positive, Negative, Neutral) specifically targeting the extracted aspect.
- **Cause Extraction (QA Pipeline):** Uses a Question-Answering model to determine *why* a user left a specific sentiment (e.g., "Why is the fan negative?" -> "It is too loud").
- **Interactive Streamlit Dashboard:** 
  - **Single Inference:** Test individual reviews in real-time.
  - **Batch Comparison:** Upload CSVs of laptop reviews and generate a head-to-head professional Radar Chart and report based on user personas (Gamer, Student, Content Creator).
- **GPU Accelerated:** Automatically leverages NVIDIA CUDA or Apple Silicon (MPS) for lightning-fast batch inference.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/YourRepoName.git
   cd YourRepoName
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Model Weights:**
   Because of GitHub's file size limits, the trained `.pth` model weights are hosted in the **Releases** tab of this repository.
   - Download `roberta.pth` (Sentiment Model)
   - Download `bert_ae_best_model.pth` (Aspect Extraction Model)
   - Place both files directly into the root folder of this project.

## 💻 Running the App

Once the dependencies are installed and the models are in the main folder, you can start the dashboard:

```bash
streamlit run final.py
```

## 📁 Repository Structure
- `final.py` - The main Streamlit application and NLP inference pipeline.
- `colab.ipynb` - The Google Colab notebook used to train and evaluate the models.
- `Laptop_Train_v2.csv` - The primary training dataset used for the models.
- `laptop_datasets_10/` & `laptops_dataset_final_600.csv` - Sample datasets for testing the batch comparison feature.
