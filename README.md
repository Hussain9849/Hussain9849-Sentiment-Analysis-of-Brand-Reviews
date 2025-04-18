# Sentiment Analysis of Brand Reviews

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Overview
This project is a **Sentiment Analysis Tool** designed to analyze customer feedback and reviews. It provides insights into customer sentiments, helping businesses improve their products and services. The tool supports individual review analysis, CSV file uploads, and e-commerce product reviews.

### 🌟 Features
- **Individual Review Analysis**: Analyze the sentiment of a single review.
- **CSV File Analysis**: Upload a CSV file to analyze multiple reviews at once.
- **E-commerce Reviews**: Fetch and analyze reviews for e-commerce products.
- **Aspect-Based Sentiment Analysis**: Break down sentiments by aspects like quality, price, and customer service.
- **Interactive Visualizations**: Polarity distribution, word clouds, and pie charts.
- **Insights and Recommendations**: Actionable insights based on sentiment analysis.




## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Hussain9849/Hussain9849-Sentiment-Analysis-of-Brand-Reviews.git
   cd Hussain9849-Sentiment-Analysis-of-Brand-Reviews
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run sentiment_analysis.py
   ```

---

## 📂 Project Structure
```
📦 Sentiment-Analysis-of-Brand-Reviews
├── [`Demo_Project/sentiment_analysis.py`](Demo_Project/sentiment_analysis.py )       # Main application file
├── styles.css                  # Custom CSS for styling
├── products.json               # Sample e-commerce product data
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── assets/                     # Images and other assets
```

---

## ⚙️ Features in Detail

### 1. **Individual Review Analysis**
- Enter a single review and get:
  - **Polarity**: Positive, Negative, or Neutral.
  - **Subjectivity**: Opinion-based or fact-based.

### 2. **CSV File Analysis**
- Upload a CSV file containing reviews.
- Get detailed sentiment analysis for each review.
- Download the results as a CSV file.

### 3. **E-commerce Reviews**
- Fetch reviews for products from a JSON file.
- Analyze reviews for specific products.

### 4. **Visualizations**
- **Word Cloud**: Most frequently used words in reviews.
- **Polarity Distribution**: Histogram of sentiment scores.
- **Aspect-Based Sentiment Analysis**: Pie charts for aspects like quality, price, and customer service.

---

## 📊 Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Libraries**:
  - `TextBlob` for sentiment analysis
  - `cleantext` for text preprocessing
  - `Plotly` for interactive visualizations
  - `WordCloud` for generating word clouds
  - `Pandas` for data manipulation

---

## 🤝 Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## 📬 Contact
For any questions or feedback, feel free to reach out:
- **GitHub**: [Hussain9849](https://github.com/Hussain9849)
- **Email**: mahammadhussainshaik54@gmail.com

---

## 🌟 Acknowledgements
- [Streamlit](https://streamlit.io/) for the amazing framework.
- [TextBlob](https://textblob.readthedocs.io/en/dev/) for sentiment analysis.
- [Plotly](https://plotly.com/) for interactive visualizations.
- [WordCloud](https://github.com/amueller/word_cloud) for word cloud generation.

---

## 📈 Future Enhancements
- Integrate with live e-commerce APIs (e.g., Amazon, Flipkart).
- Add multilingual sentiment analysis.
- Implement advanced NLP techniques using transformers (e.g., BERT).
- Add a dashboard for real-time sentiment monitoring.

