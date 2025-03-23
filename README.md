# Spam SMS Detection

## Overview
This project is a **Spam SMS Detection Model** that classifies SMS messages as either **spam** or **ham (not spam)** using **Machine Learning** techniques. The model utilizes **TF-IDF vectorization** and a **Naïve Bayes classifier** to achieve high accuracy.

## Dataset
The dataset consists of labeled SMS messages categorized as spam or non-spam (ham). The data is preprocessed to remove special characters and stopwords before applying feature extraction.

## Features
- **Text Preprocessing**: Tokenization, stopword removal, and vectorization.
- **TF-IDF Vectorization**: Converts text messages into numerical features.
- **Naïve Bayes Classification**: A probabilistic model optimized for text classification.
- **Performance Metrics**: Evaluates accuracy, precision, recall, and F1-score.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/spam-sms-detection.git
   ```
2. Navigate to the project folder:
   ```sh
   cd spam-sms-detection
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the model with the following command:
```sh
python spam_detection.py
```

## Dependencies
- Python 3.x
- Pandas
- Scikit-learn
- Numpy

## Model Performance
The model achieves an accuracy of **96.68%**, with strong precision and recall scores.

## Future Improvements
- Implement deep learning models (LSTMs, Transformers) for better accuracy.
- Deploy as a web application for real-time spam detection.

## Contributing
Feel free to fork this repository and submit pull requests with improvements!

## License
This project is licensed under the MIT License.


