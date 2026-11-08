# Car Price Prediction with Machine Learning

Predicts the selling price of a used vehicle from its new price, age, kilometres driven, fuel type, transmission, seller type, owner count and brand. Built as part of the CodeAlpha Data Science internship task *"Car Price Prediction with Machine Learning"*.

Everything (inspection, cleaning, feature engineering, EDA, model comparison, tuning, evaluation, error analysis) is in one notebook: [`car_price_prediction.ipynb`](car_price_prediction.ipynb).

## Dataset

`data/car_data.csv`: 301 listings (299 after removing 2 exact duplicates), 9 columns: car name, year, selling price, present (new) price, kilometres driven, fuel type, seller type, transmission, owner count. Prices are assumed to be in lakh INR (1 lakh = 100,000 INR).

Two things worth knowing about the data:
- It has **no horsepower and no brand column**. Mileage is available only as kilometres driven. Brand and vehicle type were derived from the car name.
- It mixes **198 cars and 101 motorcycles/scooters**, which sell for about ten times less.

## Approach

1. **Inspect and clean:** check types, missing values and duplicates; remove 2 duplicate rows. Outliers (Land Cruiser at 35 lakh, one vehicle with 500,000 km) are kept because they are real listings.
2. **Feature engineering:** `car_age` (from the year), `brand` and `vehicle_type` (from the car name; rare brands grouped as `Other`). Categorical columns are one-hot encoded inside a scikit-learn pipeline, so there is no leakage from the test data.
3. **Model comparison:** Linear Regression, Ridge, Random Forest and Gradient Boosting, compared with repeated 5-fold cross-validation on the training data. A 20% hold-out set is kept for the final test.
4. **Tuning and evaluation:** grid search on the two best tree models, a single final evaluation on the hold-out set, error analysis, and a stability check over 20 random splits.
5. **Interpretation:** permutation importance and a prediction helper for new vehicles.

## Results

| Model (5-fold CV x 3, training data) | R2 | MAE (lakh) | RMSE (lakh) |
|---|---|---|---|
| Random Forest | 0.890 | 0.671 | 1.512 |
| Gradient Boosting | 0.877 | 0.684 | 1.579 |
| Ridge | 0.863 | 1.150 | 1.784 |
| Linear Regression | 0.859 | 1.170 | 1.810 |

Always predicting the mean price gives an MAE of 3.4 lakh. After tuning, Gradient Boosting and Random Forest are practically tied (CV MAE 0.669 vs 0.675); Gradient Boosting was selected.

**Hold-out test (60 vehicles):** R2 0.823, MAE 1.01 lakh. **Across 20 random splits:** R2 averages 0.928 (range 0.778 to 0.972), MAE averages 0.575 lakh.

![Depreciation, fuel type and transmission](outputs/figures/02_depreciation_fuel_transmission.png)
![Actual vs predicted](outputs/figures/03_actual_vs_predicted.png)
![Feature importance](outputs/figures/04_feature_importance.png)

### Key findings

- **New price and age drive the resale price.** Shuffling `Present_Price` raises the error by about 3.6 lakh, `car_age` by about 1.0 lakh; the other features matter little. Cars keep roughly 82% of their new price at 0-2 years, 66% at 3-5 years, 47% at 6-8 years and 26% at 9+ years.
- **Tree models beat linear models** (error about 0.67 vs 1.15 lakh) because depreciation is a percentage that changes with age, which is not a straight-line relationship.
- **Feature engineering did not measurably improve accuracy.** The cross-validated error stays at about 0.67 lakh with or without `car_age`, `vehicle_type` and `brand`, because the new price already carries that information.
- **Diesel and automatic cars sell for more, but this is mostly because they are the more expensive models**, not because of the fuel or gearbox itself (their importance is close to zero once the new price is known).
- **Where the model fails:** a few expensive and old vehicles. On the hold-out set, the 5 worst predictions account for 88% of the total squared error; for example, a 12-year-old Camry (142,000 km) that sold for 2.5 lakh was predicted at 10.6 lakh. Vehicles under 2 lakh have a mean error of only 0.24 lakh.

## Limitations

- Small dataset: 299 rows and only 12 vehicles above 15 lakh. Predictions for expensive or old cars are unreliable.
- The same car models appear in both training and test data, so accuracy on a model the system has never seen may be lower. Grouped cross-validation by car name would measure this.
- The model needs the **new price of that exact model** as an input.
- No horsepower, condition or location; the data ends in 2018, so it should not be used for current prices without retraining.
- The 20-split scores are slightly optimistic because hyperparameters were tuned on data that overlaps with those splits. A realistic typical error is 0.6-0.7 lakh.

## Project structure

```
car-price-prediction/
├── car_price_prediction.ipynb   # full analysis with outputs
├── data/car_data.csv
├── outputs/figures/             # charts used in this README
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook car_price_prediction.ipynb
```

Start Jupyter from the project root folder, then run all cells. Results can differ slightly with other library versions.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Jupyter.

## Author

Anas Ali Tayyab, BS Data Science student, The Islamia University of Bahawalpur.
