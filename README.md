# Movie Rating Prediction

This project involves predicting a movie's rating based on features such as genre, director, and actors. Below is a detailed explanation of the steps taken to complete the task.

## Dataset
The dataset contains information about movies, including columns like `Genre`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`, and `Rating`. The `Rating` column is the target variable that we aim to predict.

## Steps

### 1. Importing Libraries
We start by importing necessary Python libraries for data analysis, visualization, and machine learning:
- `pandas` and `numpy` for data handling.
- `matplotlib` and `seaborn` for data visualization.
- `sklearn` for preprocessing, model building, and evaluation.

### 2. Loading the Dataset
The dataset is loaded into a Pandas DataFrame using the `pd.read_csv()` function. The file encoding was set to `ISO-8859-1` to handle special characters in the data.

### 3. Data Preprocessing
- **Checking for Missing Values**: We identified missing values in the dataset and dropped rows where the `Rating` column was missing.
- **Feature Selection**: Relevant features (`Genre`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`) were selected for building the model.

### 4. Exploratory Data Analysis (EDA)
- **Summary Statistics**: We generated descriptive statistics to understand the distribution of numerical features.
- **Visualization**:
  - A histogram was plotted to visualize the distribution of movie ratings.
  - A correlation heatmap was created to analyze relationships between numerical features.

### 5. Splitting the Data
The dataset was split into training and testing sets using an 80-20 ratio with the `train_test_split` function from `sklearn`.

### 6. Preprocessing Pipeline
A `ColumnTransformer` was used to preprocess categorical features (`Genre`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`). The `OneHotEncoder` was applied to convert these categorical variables into numerical format suitable for the model.

### 7. Model Building
A `RandomForestRegressor` was chosen as the machine learning model due to its robustness and ability to handle mixed data types. The model was wrapped in a pipeline to include preprocessing and training in a single step.

### 8. Training and Prediction
The model was trained on the training data, and predictions were made on the test set.

### 9. Evaluation
The model's performance was evaluated using:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted ratings.
- **R-squared Score (R²)**: Indicates the proportion of variance explained by the model.

### 10. Saving Predictions
The actual and predicted ratings were saved to a CSV file (`movie_rating_predictions.csv`) for further analysis.

## Results
- **Mean Squared Error (MSE)**: 1.4648
- **R-squared Score (R²)**: 0.2121

## How to Run the Code
1. Clone the repository and ensure the dataset file (`IMDb Movies India.csv`) is in the correct location.
2. Open the Python file (`movie_rating_prediction.py`) in your IDE or run it in Google Colab.
3. Install required libraries if not already installed.
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Update the dataset path (`file_path`) in the code if needed.
5. Run the script to train the model and generate predictions.
6. Check the `movie_rating_predictions.csv` file for the output.

## Challenges
- Handling missing values in the dataset.
- Preprocessing categorical features for machine learning.
- Selecting a model that balances simplicity and performance.

## Future Improvements
- Fine-tune the model hyperparameters for better accuracy.
- Include additional features like movie budget, box office collections, or reviews.
- Experiment with other regression algorithms like XGBoost or LightGBM.

## Dependencies
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Feel free to use this code and improve upon it. If you encounter any issues, please reach out!
