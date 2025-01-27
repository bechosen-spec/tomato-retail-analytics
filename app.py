import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Tomato Retail Analytics", layout="wide")

# Load Data
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            required_columns = ['Supplier Cost (NGN)', 'Restocked Quantity (kg)', 'Transport Cost (NGN)', 'Total Sales Value (NGN)']
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
options = [
    "Home", "Business Analytics", "Market Forecast/Stock Tracking", "Educative Tips", "Explanatory/Statistical Analysis"
]
selected_option = st.sidebar.selectbox("Choose a section:", options)

# Home Section
if selected_option == "Home":
    st.title("Welcome to Tomato Retail Analytics")
    st.write("This app provides an interactive way to explore and analyze retail data for tomato sales.")
    st.image("tomatoes image.jpg", caption="Tomato Analytics", use_container_width=True)
    st.write("Navigate through the sections to discover insights about trends, supplier performance, location-based analysis, and more.")

# Business Analytics Section
elif selected_option == "Business Analytics":
    sub_options = ["Trend Analysis", "Supplier Analysis", "Location Analysis", "Seasonal Analysis"]
    selected_sub_option = st.sidebar.selectbox("Select Analysis Type:", sub_options)

    if selected_sub_option == "Trend Analysis" and data is not None:
        st.title("Trend Analysis")
        st.write("Explore yearly trends in profit.")
        annual_profit = data['Profit'].resample('A').mean()
        st.line_chart(annual_profit)
        st.write(f"**Mean Profit:** {annual_profit.mean():.2f}")
        st.write(f"**Standard Deviation:** {annual_profit.std():.2f}")

    elif selected_sub_option == "Supplier Analysis" and data is not None:
        st.title("Supplier Analysis")
        st.write("Analyze the average costs associated with each supplier.")
        supplier_cost = original_data.groupby('Supplier Name')['Supplier Cost (NGN)'].mean().sort_values()
        st.bar_chart(supplier_cost)
        st.write("The chart above shows the average supplier costs.")

    elif selected_sub_option == "Location Analysis" and data is not None:
        st.title("Location Analysis")
        st.write("Examine performance across different store locations.")
        location_performance = original_data.groupby('Store Location')[['Quantity Sold (kg)', 'Total Sales Value (NGN)']].agg(['sum', 'mean'])
        st.dataframe(location_performance)
        fig, ax = plt.subplots(figsize=(12, 6))
        location_performance[('Quantity Sold (kg)', 'sum')].sort_values(ascending=False).plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Total Quantity Sold by Store Location')
        ax.set_ylabel('Quantity Sold (kg)')
        ax.set_xlabel('Store Location')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif selected_sub_option == "Seasonal Analysis" and data is not None:
        st.title("Seasonal Analysis")
        st.write("Understand how sales vary across different seasons.")
        seasonal_quantity = original_data.groupby('Season')['Quantity Sold (kg)'].sum()
        st.bar_chart(seasonal_quantity)

# Market Forecast/Stock Tracking Section
elif selected_option == "Market Forecast/Stock Tracking":
    sub_options = ["Price Strategy Recommendation", "Stock Forecast & Tracking"]
    selected_sub_option = st.sidebar.selectbox("Select Forecast Type:", sub_options)

    if selected_sub_option == "Price Strategy Recommendation" and data is not None:
        st.title("Price Strategy Recommendation")
        st.write("Explore correlations between key numeric variables to inform pricing strategies.")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)

    elif selected_sub_option == "Stock Forecast & Tracking" and data is not None:
        st.title("Stock Forecast & Tracking")
        st.write("Predict future stock requirements.")
        selected_date = st.date_input("Select a date for prediction:")
        st.write(f"Selected Date: {selected_date}")

        features = ['Restocked Quantity (kg)', 'Unit Price (NGN)', 'Opening Stock (kg)']
        target = 'Quantity Sold (kg)'

        if all(feature in data.columns for feature in features):
            X = data[features]
            y = data[target]

            # Split data and train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Prediction for test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            st.write(f"**Error Margin (Lower is Better):** {rmse:.2f}")

            # Prediction for selected date
            if not data.empty:
                sample_features = X.mean().to_frame().T  # Default prediction with average features
                prediction = model.predict(sample_features)[0]
                st.write(f"**Predicted Quantity Sold for {selected_date}:** {prediction:.2f} kg")
        else:
            st.warning("Required features are missing in the dataset for prediction.")

# Educative Tips Section
elif selected_option == "Educative Tips":
    st.title("Educative Tips")
    st.write("Explore helpful resources and guides related to tomato cultivation and sales.")
    tips = [
        {"title": "Guide to Tomato Cultivation in Nigeria: Tips, Techniques & Insights", "url": "https://htsfarms.ng/2024/03/20/guide-to-tomato-cultivation-in-nigeria-tips-techniques-insights/"},
        {"title": "Tomatoes Provide a Great Boost to Health", "url": "https://www.uaex.uada.edu/counties/miller/news/fcs/fruits-veggies/Tomatoes_Provide_a_Great_Boost_to_Health.aspx"},
        {"title": "Tomato Farming in Nigeria", "url": "https://eniolafarms.wordpress.com/2018/11/24/tomato-farming-in-nigeria/"},
        {"title": "Growing Tomatoes: How to Plant, Maintain, and Harvest", "url": "https://eos.com/blog/how-to-grow-tomatoes/#:~:text=Pruning.%20If%20you%20want%20your%20plants%20to,removing%20suckers,%20low-hanging%20branches,%20and%20wilted%20leaves."},
        {"title": "Tomato Plant Care: How to Have More Productive Tomato Plants", "url": "https://grow.ifa.coop/gardening/guide-to-tomato-plant-care"},
        {"title": "The Ultimate Guide to Tomato Growing", "url": "https://www.lovethegarden.com/au-en/growing-guide/how-grow-tomatoes#:~:text=Tomato%20plant%20care,deprive%20the%20roots%20of%20oxygen."}
    ]
    for tip in tips:
        st.write(f"- [{tip['title']}]({tip['url']})")

# Explanatory/Statistical Analysis Section
elif selected_option == "Explanatory/Statistical Analysis" and data is not None:
    st.title("Explanatory/Statistical Analysis")
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
