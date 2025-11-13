import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_sleep_data(df):
    df = df.copy()

    # Handle missing values
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No Disorder')

    # Pisahkan Sistol & Diastol
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['BP Sistol'] = bp_split[0].astype(float)
    df['BP Diastol'] = bp_split[1].astype(float)
    df = df.drop(columns=['Blood Pressure'])

    df = df.drop(columns=['Person ID'])

    # Samakan kategori BMI
    df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})

    # Encoding
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    df = pd.get_dummies(df, columns=['Occupation'], drop_first=True)

    bmi_order = [['Overweight', 'Normal', 'Obese']]
    encoder = OrdinalEncoder(categories=bmi_order)
    df['BMI Category'] = encoder.fit_transform(df[['BMI Category']])

    # Split fitur & target
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisasi fitur numerik
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Gabung dan simpan 
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return df_train, df_test

if __name__ == "__main__":
    os.makedirs("Sleep_health_and_lifestyle_dataset_preprocessing", exist_ok=True)
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

    df_train, df_test = preprocess_sleep_data(df)

    df_train.to_csv("preprocessing/Sleep_health_and_lifestyle_dataset_preprocessing/Sleep_health_and_lifestyle_dataset_train.csv", index=False)
    df_test.to_csv("preprocessing/Sleep_health_and_lifestyle_dataset_preprocessing/Sleep_health_and_lifestyle_dataset_test.csv", index=False)
