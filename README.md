# Tomato Retail Analytics

## Overview
**Tomato Retail Analytics** is a Streamlit-based web application designed to help users analyze and visualize retail data for tomato sales. The app provides insights into trends, supplier performance, location-based sales, seasonal patterns, and stock forecasting. This tool is ideal for businesses, analysts, and decision-makers looking to optimize operations and improve profitability.

## Features
The application is divided into multiple sections for user convenience:

1. **Home**
   - Introduction to the app with an image and navigation details.

2. **Trend Analysis**
   - Yearly trend visualization of profits.
   - Statistical metrics such as mean and standard deviation.

3. **Supplier Analysis**
   - Evaluation of average supplier costs.
   - Bar chart representation of supplier performance.

4. **Location Analysis**
   - Aggregated sales performance across different store locations.
   - Bar chart visualization of total quantity sold by location.

5. **Seasonal Analysis**
   - Identification of sales trends across different seasons.
   - Seasonal quantity sold visualization.

6. **Price Strategy & Recommendations**
   - Correlation heatmap of key numeric variables to identify pricing strategies.

7. **Stock Forecast & Tracking**
   - Predictive modeling for stock requirements using Random Forest regression.
   - RMSE calculation for model performance evaluation.

8. **Exploratory Analysis**
   - Descriptive statistics and correlation heatmaps for in-depth data exploration.

## Setup and Installation
Follow these steps to set up and run the application:

### Prerequisites
- Python 3.8 or higher installed.
- Basic understanding of Python and Streamlit.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tomato-retail-analytics.git
   cd tomato-retail-analytics
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open the provided URL in your browser to access the app.

## Usage
### Upload Data
1. Prepare your CSV dataset containing the following required columns:
   - `Date`
   - `Supplier Cost (NGN)`
   - `Restocked Quantity (kg)`
   - `Transport Cost (NGN)`
   - `Total Sales Value (NGN)`
   - Additional columns for location and seasonal analysis.

2. Upload the dataset using the file uploader in the sidebar.

### Navigate Sections
- Use the sidebar navigation to explore different sections of the app.
- Each section provides specific insights, visualizations, or recommendations.

## Project Structure
```
.
├── app.py                 # Main Streamlit application script
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
├── data/                  # Directory for sample datasets
└── images/                # Directory for images used in the app
```

## Key Dependencies
- `streamlit`: For building the web interface.
- `pandas`: For data manipulation.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For predictive modeling.

## Screenshots
### Home Page
![Home Page](images/home_page.png)

### Trend Analysis
![Trend Analysis](images/trend_analysis.png)

### Location Analysis
![Location Analysis](images/location_analysis.png)

## Future Improvements
- Add advanced forecasting models.
- Enhance visualizations with interactive elements.
- Implement user authentication for saving analyses.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For any questions or suggestions, please contact:
- **Name**: Christian Nwalu
- **Email**: christian@example.com
- **GitHub**: [your-username](https://github.com/bechosen-spec/tomato-retail-analytics)

