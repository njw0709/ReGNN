import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    "ID": [1, 2, 3, 4, 5, 6],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],  # Binary
    "Response": ["Yes", "No", "Yes", "No", "Yes", "Yes"],  # Binary
    "City": ["New York", "London", "Paris", "New York", "London", "Tokyo"],  # Nominal
    "Education": [
        "High School",
        "Bachelor",
        "Master",
        "Bachelor",
        "High School",
        "PhD",
    ],  # Ordinal (can be treated as nominal too)
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# --- Method 1: Binary Encoding using map() ---
print("\n--- Binary Encoding with map() ---")
df_mapped = df.copy()
df_mapped["Gender_Mapped"] = df_mapped["Gender"].map({"Male": 0, "Female": 1})
df_mapped["Response_Mapped"] = df_mapped["Response"].map({"Yes": 1, "No": 0})
print(df_mapped[["Gender", "Gender_Mapped", "Response", "Response_Mapped"]])

# --- Method 2: Nominal Encoding with astype('category').cat.codes ---
print("\n--- Nominal Encoding with cat.codes ---")
df_cat_codes = df.copy()
df_cat_codes["City_Codes"] = df_cat_codes["City"].astype("category").cat.codes
# Display unique cities and their codes for clarity
print(
    "City mapping:",
    dict(enumerate(df_cat_codes["City"].astype("category").cat.categories)),
)
print(df_cat_codes[["City", "City_Codes"]])

# --- Method 3: Nominal Encoding with pd.factorize() ---
print("\n--- Nominal Encoding with pd.factorize() ---")
df_factorized = df.copy()
codes, uniques = pd.factorize(df_factorized["City"])
df_factorized["City_Factorized"] = codes
print("Factorized uniques:", uniques)
print(df_factorized[["City", "City_Factorized"]])

# --- Method 4: One-Hot Encoding with pd.get_dummies() ---
print("\n--- One-Hot Encoding with pd.get_dummies() ---")
df_one_hot = df.copy()
# For binary columns, drop_first=True gives a single 0/1 column
df_one_hot = pd.get_dummies(
    df_one_hot, columns=["Gender"], prefix="Gender", drop_first=True
)
# For multi-category nominal columns
df_one_hot = pd.get_dummies(df_one_hot, columns=["City"], prefix="City")
print(df_one_hot.filter(like="Gender_").join(df_one_hot.filter(like="City_")))

# --- Method 5: Ordinal Encoding with map() (if order is known) ---
print("\n--- Ordinal Encoding with map() ---")
df_ordinal = df.copy()
education_ranking = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
df_ordinal["Education_Ordinal"] = df_ordinal["Education"].map(education_ranking)
print(df_ordinal[["Education", "Education_Ordinal"]])

print("\nNote: For 'fast' in terms of performance, ")
print(
    "`.astype('category').cat.codes` and `pd.factorize()` are generally very efficient for integer encoding."
)
print("`pd.get_dummies()` is also highly optimized for one-hot encoding.")
