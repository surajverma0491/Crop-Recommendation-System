# Crop Recommendation System

This project aims to develop a crop recommendation system based on various environmental and soil factors. The goal is to predict the most suitable crop for a given set of conditions using machine learning techniques.

## Project Overview

The project uses a dataset of different crops and their associated environmental and soil features (e.g., Nitrogen, Phosphorus, Potassium content, temperature, humidity, pH, and rainfall). The model predicts the most suitable crop based on these parameters. Several machine learning algorithms are tested to find the best-performing model, and the final model is saved for future predictions.

## Dataset

The dataset used in this project is sourced from Kaggle, with crop information including:
- N (Nitrogen)
- P (Phosphorus)
- K (Potassium)
- Temperature
- Humidity
- pH
- Rainfall

## Steps Involved

1. **Data Preprocessing**:
   - Loading and exploring the dataset.
   - Handling missing values and duplicates.
   - Descriptive statistics and correlation analysis.
   - Visualizing the distributions of nutrients and features.

2. **Feature Encoding**:
   - Mapping crop names to numerical labels using a dictionary for easy classification.

3. **Data Splitting**:
   - Splitting the dataset into training and testing sets.
   - Scaling the features using MinMaxScaler and StandardScaler.

4. **Modeling**:
   - Training multiple machine learning models:
     - Logistic Regression
     - Naive Bayes
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Random Forest Classifier
     - AdaBoost Classifier
     - Gradient Boosting Classifier
   - Evaluating each model's performance using accuracy.

5. **Recommendation System**:
   - Using the best-performing model (Random Forest Classifier) to predict suitable crops based on user input (N, P, K, temperature, humidity, pH, rainfall).

6. **Model Saving**:
   - Saving the trained model and scalers (MinMaxScaler and StandardScaler) using pickle for future use.


## File Structure

```plaintext
.
├── dataset_kaggle.csv        # Dataset with crop and environmental data
├── model.pkl                 # Saved machine learning model (Random Forest)
├── minmaxscaler.pkl          # Saved MinMaxScaler
├── standscaler.pkl           # Saved StandardScaler
├── README.md                 # Project overview and instructions
└── crop_recommendation.py    # Python script with the full implementation

```

## Usage

### Requirements

- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- Seaborn
- Matplotlib
- Pickle

### Installation

To run this project, ensure you have the necessary libraries installed. You can install the required dependencies using `pip`:
pip install pandas numpy scikit-learn seaborn matplotlib

### Running the System

1. **Training the Model**:
   To train the model, run the `crop_recommendation.py` script, which will load the dataset, train the models, and display the results.

2. **Making Predictions**:
   After training the model, you can use the `recommendation` function to make predictions for a specific set of input features:

   ```python
   N = 90
   P = 42
   K = 43
   temperature = 20.879744
   humidity = 82.002744
   ph = 6.502985
   rainfall = 202.935536

   prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)
   print(f"The recommended crop is: {prediction}")

   ```
![Initiated_Image](https://drive.google.com/file/d/1EF8kHf7fE3ekgdpARXTRsCBGwA1zHw42/view?usp=sharing)

   ### 3. Loading the Saved Model: You can load the saved model and scalers using pickle:
   model = pickle.load(open('model.pkl', 'rb'))
   mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
   sc = pickle.load(open('standscaler.pkl', 'rb'))

## Conclusion
This crop recommendation system leverages machine learning techniques to predict suitable crops based on environmental and soil conditions. The model is optimized using multiple classifiers and then used for real-time predictions. The system is saved and ready for future use, providing a valuable tool for farmers and agricultural experts.

## License
This project is open-source and available under the MIT License.
