# 📱 Spam SMS Detection using Machine Learning

This project is a machine learning-based solution to classify SMS messages as **Spam** or **Ham (Not Spam)**. The model uses NLP techniques and classification algorithms to detect spam messages with high accuracy.

---

## 🎯 Task Objectives

- Preprocess and clean SMS text data.
- Convert text to numerical representation using TF-IDF.
- Train and evaluate classification models such as Naive Bayes and SVM.
- Visualize data and model performance using charts and confusion matrix.
- Provide clear evaluation metrics.

---

## 🚀 How to Run the Project

### 1. Clone the repository

    git clone https://github.com/jatin019/spam-sms-detection.git
    cd spam-sms-detection

### 2. Install the dependencies

    pip install -r requirements.txt

Or manually install them:

    pip install numpy pandas scikit-learn matplotlib seaborn nltk

### 3. Run the Jupyter Notebook

    jupyter notebook "Spam SMS Detection.ipynb"

Follow the notebook cells step-by-step to preprocess data, train models, and view results.

---

## 🧠 Models Implemented

- **Multinomial Naive Bayes** – Effective for text classification.
- **Support Vector Machine (SVM)** – Robust against high-dimensional text data.
- **Logistic Regression** – Simple yet powerful linear model for binary classification.
- **Random Forest Classifier** – Ensemble model that reduces overfitting and improves accuracy.


---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

These metrics help evaluate how well the model distinguishes between spam and ham.

---


## 🧹 Code Quality

- Well-commented and modular code
- Structured flow: loading → cleaning → modeling → evaluating
- Logical separation of tasks into notebook cells

---

## 🌟 Innovation & Creativity

- Preprocessing with NLTK for stopwords and tokenization
- TF-IDF vectorization to enhance feature representation
- Multiple classifiers for comparative performance
- Visual insights into spam vs. ham word usage

---

## 📂 Project Structure

    spam-sms-detection/
    ├── Spam SMS Detection.ipynb
    ├── README.md
    ├── requirements.txt
    └── dataset/               

---

## 📝 Dataset

This project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains 5,574 labeled SMS messages.

**Note:** Download the dataset and place it in the `dataset/` folder.

---

## 🛠 Dependencies

Main libraries used:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- nltk

See `requirements.txt` for details.

---

## 📬 Contact

For questions or contributions, feel free to reach out or open an issue on GitHub.

---

## 📘 License

This project is open-source under the [MIT License](LICENSE).
