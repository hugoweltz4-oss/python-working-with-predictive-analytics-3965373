# %%

# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values

# TODO: Check how many values are missing (NaN)
print("How many values are missing ?")
print(data.isnull().sum())

# Option 1: Drop the entire column with missing values
# TODO: Add code to drop the 'bmi' column and verify
data_option1 = data.copy()
data_option1.drop("bmi", axis=1, inplace=True)
print("\nOption1 : Drop the 'bmi' column")
print(data_option1.isnull().sum())

# Option 2: Drop rows with missing values
# TODO: Add code to drop rows with missing values and verify
data_option2 = data.copy()
data_option2.dropna(inplace=True)
print("\nOption2 : Drop rows with missing value")
print(data_option2.isnull().sum())

# Option 3: Fill missing values with mean (SimpleImputer)
# TODO: Add code to fill missing values in the 'bmi' column using SimpleImputer
data_option3 = data.copy()
imputer = SimpleImputer(strategy="mean")
data_option3["bmi"] = imputer.fit_transform(data_option3[["bmi"]])
print("\nOption3 : Fill missing values with mean (SimpleImputer)")
print(data_option3.isnull().sum())

# Visualization

# TODO: Create a scatterplot (Age vs. Charges)
plt.figure()
plt.plot(data["age"], data["charges"])
plt.show()


# TODO: Create a correlation heatmap
