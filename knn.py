import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st

# Ignore all warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("trip_time_dataset_1000 (1).csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df)

X= df.loc[:, 'Distance_km':'Weather']
y = df['Trip_Time_min']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from sklearn.neighbors import KNeighborsRegressor

regressor=KNeighborsRegressor(n_neighbors=6,algorithm='auto')
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))




# Title of the app
st.title("🚀 Travel Mitre")

st.subheader("Enter the following feature values:")

# Input fields
distance_km = st.number_input("Distance_km:", min_value=0, step=1)
travel_mode = st.selectbox("Travel_mode:", [0, 1, 2, 3], help="0=Car, 1=Train, 2=Flight, 3=Bus")
avg_speed = st.number_input("Avg_Speed_kmph:", min_value=0, step=1)
day_time = st.slider("Day_Time (1-24):", min_value=1, max_value=24, value=7)
weather = st.number_input("Weather:", min_value=0, step=1)

# Predict button
if st.button("Predict Travel Time"):
    if avg_speed > 0:
        # Prepare input for model (must match training features order!)
        input_features = np.array([[distance_km, travel_mode, avg_speed, day_time, weather]])
        
        # ML prediction
        predicted_time = regressor.predict(input_features)[0]
        
        st.success(f"🕒 Predicted Travel Time: {predicted_time:.2f} minutes")
    else:
        st.error("⚠️ Avg speed must be greater than 0")


import matplotlib.pyplot as plt

st.subheader("📊 Model Performance")

# Plot predicted vs actual
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Trip Time (min)")
ax.set_ylabel("Predicted Trip Time (min)")
ax.set_title("Predicted vs Actual Travel Times")

st.pyplot(fig)
