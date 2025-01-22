import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set page configuration
st.set_page_config(page_title="Tomato Retail Analytics", layout="wide")

# Load Data
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            required_columns = ['Supplier Cost (NGN)', 'Restocked Quantity (kg)', 'Transport Cost (NGN)']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' is missing from the dataset.")
            df['Total Supply Cost (NGN)'] = (df['Supplier Cost (NGN)'] * df['Restocked Quantity (kg)']) + df['Transport Cost (NGN)']
            df['Profit'] = df['Total Sales Value (NGN)'] - df['Total Supply Cost (NGN)']
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        st.warning("Please upload a dataset to proceed.")
        return None

data = load_data()
if data is not None:
    original_data = data.copy()
    data = pd.get_dummies(data, drop_first=True)

# Sidebar navigation
st.sidebar.title("Navigation")
options = ["Home", "Trend Analysis", "Supplier Analysis", "Location Analysis", "Seasonal Analysis", "Price Strategy & Recommendations", "Stock Forecast & Tracking", "Exploratory Analysis"]
selected_option = st.sidebar.radio("Choose a section:", options)

# Home Section
if selected_option == "Home":
    st.title("Welcome to Tomato Retail Analytics")
    st.write("This app provides an interactive way to explore and analyze retail data for tomato sales.")
    st.image("/home/oem/tomato-retail-analytics/tomatoes image.jpg", caption="Tomato Analytics", use_container_width=True)
    st.write("Navigate through the sections to discover insights about trends, supplier performance, location-based analysis, and more.")

# Trend Analysis Section
elif selected_option == "Trend Analysis" and data is not None:
    st.title("Trend Analysis")
    st.write("Explore yearly trends in profit.")
    annual_profit = data['Profit'].resample('A').mean()
    st.line_chart(annual_profit)
    mean_profit = annual_profit.mean()
    std_profit = annual_profit.std()
    st.write(f"**Mean Profit:** {mean_profit:.2f}")
    st.write(f"**Standard Deviation:** {std_profit:.2f}")

# Supplier Analysis Section
elif selected_option == "Supplier Analysis" and data is not None:
    st.title("Supplier Analysis")
    st.write("Analyze the average costs associated with each supplier.")
    supplier_cost = original_data.groupby('Supplier Name')['Supplier Cost (NGN)'].mean().sort_values()
    st.bar_chart(supplier_cost)
    st.write("The chart above shows the average supplier costs.")

# Location Analysis Section
elif selected_option == "Location Analysis" and data is not None:
    st.title("Location Analysis")
    st.write("Examine performance across different store locations.")

    location_performance = original_data.groupby('Store Location')[['Quantity Sold (kg)', 'Total Sales Value (NGN)']].agg(['sum', 'mean'])
    st.dataframe(location_performance)

    # Visualize location performance with a bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    location_performance[('Quantity Sold (kg)', 'sum')].sort_values(ascending=False).plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Total Quantity Sold by Store Location')
    ax.set_ylabel('Quantity Sold (kg)')
    ax.set_xlabel('Store Location')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Seasonal Analysis Section
elif selected_option == "Seasonal Analysis" and data is not None:
    st.title("Seasonal Analysis")
    st.write("Understand how sales vary across different seasons.")
    seasonal_quantity = original_data.groupby('Season')['Quantity Sold (kg)'].sum()
    st.bar_chart(seasonal_quantity)

# Price Strategy & Recommendations Section
elif selected_option == "Price Strategy & Recommendations" and data is not None:
    st.title("Price Strategy & Recommendations")
    st.write("Explore correlations between key numeric variables to inform pricing strategies.")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    st.write("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Stock Forecast & Tracking Section
elif selected_option == "Stock Forecast & Tracking" and data is not None:
    st.title("Stock Forecast & Tracking")
    st.write("Predict future stock requirements.")
    features = ['Restocked Quantity (kg)', 'Unit Price (NGN)', 'Opening Stock (kg)']
    target = 'Quantity Sold (kg)'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    st.write(f"**RMSE:** {rmse:.2f}")

# Exploratory Analysis Section
elif selected_option == "Exploratory Analysis" and data is not None:
    st.title("Exploratory Analysis")
    st.write("Gain deeper insights into the dataset.")
    st.write("**Descriptive Statistics**")
    st.dataframe(data.describe())
    st.write("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# No Data Uploaded
else:
    st.warning("Please upload a dataset to access this section.")

# Footer
st.sidebar.write("Developed by Christian Nwalu")
