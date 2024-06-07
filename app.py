import streamlit as st
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

path = r"C:\Users\Admin\Downloads\HPS_data.csv"
# Load dataset
data= pd.read_csv(path)

# Drop date
data = data.drop('Date', axis=1)
data.head()

# Preprocessing 
data.fillna(0, inplace=True)  # Fill missing values
X = data.drop(columns=['Maint Cost/U'])
y = data['Maint Cost/U']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit app
st.title('Maintenance Cost Optimization with Random Forest')

# Sidebar for user input
st.sidebar.header('Input Parameters')
def user_input_features():
    data = {}
    for col in X.columns:
        data[col] = st.sidebar.number_input(f'{col}', min_value=float(X[col].min()), max_value=float(X[col].max()), value=float(X[col].mean()))
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_data)

# Predict user input
prediction = model.predict(input_data)

st.subheader('Predicted Maintenance Cost')
st.write(prediction)

# Visualize feature importances
st.subheader('Feature Importances')
importances = model.feature_importances_
indices = range(len(importances))
feature_names = X.columns

fig, ax = plt.subplots()
sns.barplot(x=importances, y=feature_names, ax=ax)
st.pyplot(fig)


