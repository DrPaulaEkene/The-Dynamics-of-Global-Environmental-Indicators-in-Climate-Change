#import all necessary modules
import numpy as np                                    #package for arrays
import pandas as pd                                   #package for analysis
import seaborn as sns                                 #package for visualization
import matplotlib.pyplot as plt                       #package for visualization 
from sklearn.cluster import KMeans                    #Importing Kmean from 
from sklearn.metrics import silhouette_score          #to calculate Silhouete score
from sklearn.preprocessing import StandardScaler      #for data normalization
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


#define all necessary functions
def load_dataset(file):
    """ Load the dataset from an Excel file 
    """
    return pd.read_excel(file)

def format_headers(df):
    """ Convert header years from format 2005 [YR2005] to 2005 
    """
    df.columns = [col.split(' ')[0] 
                  if 'YR' in col 
                  else col for col in df.columns]
    return df

def replace_missing_values(df, missing_value, new_value):
    """ Replace missing values with a specified new value 
    """
    df.replace(missing_value, new_value, inplace=True)
    return df

def convert_to_numeric(df):
    """ Convert data from string to numeric where necessary 
    """
    return df.apply(pd.to_numeric, errors='ignore')


def format_numbers(df, decimal_places):
    """ Format numbers to a specified number of decimal places 
    """
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col] = df[col].round(decimal_places)
    return df

def truncate_dataset(df, number_of_rows):
    """ Truncate the dataset to a specified number of rows 
    """
    return df.head(number_of_rows)

def rename_dataframe(df, new_name):
    """ Rename the dataframe 
    """
    df.name = new_name
    return df

def process_climate_data(df):
    """
    Process the climate data to split it into separate DataFrames 
    for each indicator.

    Args:
    df (DataFrame): The original climate data DataFrame.

    Returns:
    dict: A dictionary containing two versions of DataFrames for each indicator.
          One with years as columns and one with countries as columns.
    """
    # Count distinct indicators
    distinct_indicators = df['Series Name'].nunique()

    # Split the dataset into separate DataFrames for each indicator
    dataframes_by_indicator = {}
    for indicator in df['Series Name'].unique():
        indicator_df = df[df['Series Name'] == indicator]

        # Drop unnecessary text columns
        indicator_df = indicator_df.drop(['Series Name', 'Series Code',
                                          'Country Code'], axis=1)

        # Create two versions of the DataFrame
        df_years_as_columns = indicator_df.set_index('Country Name')
        df_countries_as_columns = indicator_df.set_index('Country Name').transpose()

        # Store in dictionary
        dataframes_by_indicator[indicator] = {
            'years_as_columns': df_years_as_columns,
            'countries_as_columns': df_countries_as_columns
        }

    return distinct_indicators, dataframes_by_indicator

def kmeans_clustering_and_visualization(data, indicators, n_clusters=3):
    """
    Perform K-Means clustering on the given data and create visualizations.
    Args:
    data (DataFrame): The DataFrame containing the indicators data.
    indicators (list): List of indicator names.
    n_clusters (int): Number of clusters for K-Means.
    Returns:
    DataFrame: The DataFrame with cluster labels.
    """
    # Normalizing the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[indicators])
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_data)
    data['Cluster'] = kmeans.labels_
    
    # Visualization
    plt.figure(figsize=(15, 10))
    for i, indicator in enumerate(indicators, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x=data.index, y=indicator, hue='Cluster', 
                        data=data, palette='viridis')
        plt.title(f"{indicator} by Cluster")
        plt.xticks(rotation=45)
        plt.tight_layout()
    return data

# Function to fit a polynomial model
def fit_polynomial(x, y, degree=2):
    """
    Fits a polynomial of a given degree to the data.
    Returns the fitted polynomial function and the covariance matrix.
    """
    coeffs, cov_matrix = curve_fit(lambda t,
                                   *coeffs: np.polyval(coeffs, t), x, y,
                                   p0=[0]*degree)
    return coeffs, cov_matrix

def predict_polynomial(model, years):
    """
    Uses the fitted polynomial model to predict values for given years.
    """
    return np.polyval(model, years)

def confidence_intervals(model, cov_matrix, years, sigma=2):
    """
    Calculates the confidence intervals for the predictions.
    """
    mean = predict_polynomial(model, years)
    uncertainty = sigma * np.sqrt(np.diag(np.polyval(np.polyder(model),
                                                     years)**2 @ cov_matrix))
    lower_bounds = mean - uncertainty
    upper_bounds = mean + uncertainty
    return lower_bounds, upper_bounds

#Read excel file for Climate Data
file = 'Climate_Data_Extract_1990_2020_5Yr_expanded.xlsx'
df = load_dataset(file)

#Use my already defined functins to clean the dataset, maintain df as name
df = format_headers(df)
df = replace_missing_values(df, '..', 0)
df = convert_to_numeric(df)
df = format_numbers(df, 2)
df = truncate_dataset(df, 400)

#rename data frame- climate_data
climate_data = rename_dataframe(df, 'climate_data')

# Display the first few rows to verify changes
print(climate_data.head())

#process the climate_data
distinct_indicators, split_dataframes = process_climate_data(climate_data)

# itemize the distinct indicators
distinct_indicators, list(split_dataframes.keys())
print(distinct_indicators, split_dataframes)

# Prepare data for clustering
selected_indicators = [
    'CO2 emissions (metric tons per capita)', 
    'Renewable energy consumption (% of total final energy consumption)', 
    'Urban population (% of total population)', 
    'Forest area (% of land area)'
]


# Filter and prepare data for each selected indicator
trend_data = {}
for indicator in selected_indicators:
    indicator_df = df[df['Series Name'] == indicator].drop(['Series Name', 'Series Code', 'Country Code'], axis=1)
    indicator_df.set_index('Country Name', inplace=True)
    trend_data[indicator] = indicator_df
    
# Select the most recent year's data (2020) for each indicator
data_2020 = pd.DataFrame({indicator: trend_data[indicator].loc[:,
                                                               '2020'] 
                          for indicator in selected_indicators})
#drop non values
data_2020.dropna(inplace=True)

# Perform K-Means clustering and visualization
clustered_data = kmeans_clustering_and_visualization(data_2020,
                                                     selected_indicators)

# Merge the dataframes of the selected indicators for correlation analysis
merged_df = pd.DataFrame()
for indicator in selected_indicators:
    # Take the mean across years for each country for each indicator
    merged_df[indicator] = trend_data[indicator].mean(axis=1)

# Calculate the correlation matrix
correlation_matrix = merged_df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5)
plt.title("Correlation Matrix of Selected Climate Indicators")
plt.show()
plt.savefig("22045525.png", dpi=300)

# Select the two weakly correlated indicators for clustering
indicator1 = 'Renewable energy consumption (% of total final energy consumption)'
indicator2 = 'Urban population (% of total population)'

# Prepare the data for clustering, drop non values
cluster_data = merged_df[[indicator1, indicator2]].dropna()

# Run KMeans clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data['Cluster'] = kmeans.fit_predict(cluster_data)
cluster_labels = kmeans.fit_predict(cluster_data) 

# Plot the clusters
plt.figure(figsize=(10, 6))

# Scatter plot for data points with different colors for each cluster
scatter = plt.scatter(cluster_data[indicator1], cluster_data[indicator2],
                      c=cluster_data['Cluster'], cmap='viridis', marker='o')

# Extract cluster centers
cluster_centers = kmeans.cluster_centers_

# Scatter plot for cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, 
            alpha=0.5, marker='o', label='Cluster centers')

# Label the cluster centers
for i, center in enumerate(cluster_centers):
    plt.text(center[0], center[1], f'Center {i}',
             horizontalalignment='center', verticalalignment='center',
             color='black', fontsize=12)

# Add title and labels
plt.title(f"Cluster Analysis: {indicator1} vs. {indicator2}")
plt.xlabel(indicator1)
plt.ylabel(indicator2)

# Add a color bar, grid and legend
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.legend()
plt.gca().set_facecolor('lightgrey')

# Show the plot
plt.show()

# Add the cluster labels to your DataFrame
cluster_data['Cluster'] = cluster_labels

# Calculate the silhouette score
silhouette_avg = silhouette_score(cluster_data[[indicator1, indicator2]], cluster_labels)

print(f"The average silhouette score for 3 clusters is: {silhouette_avg}")

# Correct the aggregation function for the cluster summary table
cluster_summary = cluster_data.groupby('Cluster').agg(
    Number_of_Countries=('Cluster', 'count'),
    Average_Renewable_Energy_Consumption=(indicator1, 'mean'),
    Average_Urban_Population=(indicator2, 'mean')
).reset_index()

print(cluster_summary)

for cluster_number in range(3):  # Since we have 3 clusters (0, 1, 2)
    countries = cluster_data[cluster_data['Cluster'] == cluster_number].index.tolist()
    print(f"Countries in Cluster {cluster_number}: {countries}")


#fitting
# Extract data for the specified countries, one from each cluster
countries = ["Germany", "Sweden", "Nigeria"]

# Filter the data for these countries
filtered_data = climate_data[climate_data['Country Name'].isin(countries)]

# Check the extracted data
print(filtered_data.head(len(countries) * 2)) # Displaying two rows per country

# Filter the data to include only renewable energy consumption data
rec_data = filtered_data[filtered_data['Series Name'].str.contains("Renewableenergy consumption")]

# Drop unnecessary columns
rec_data_cleaned = rec_data.drop(columns=["Series Name", "Series Code",
                                          "Country Code"])

# Set the country name as the index
rec_data_cleaned.set_index("Country Name", inplace=True)

# Transpose and clean the data for easier analysis
rec_data_transposed = rec_data_cleaned.T

# Convert the years in the columns to datetime and ensuring the data is numeric
rec_data_transposed.index = pd.to_datetime(rec_data_transposed.index.str.extract("(\d{4})")[0])
rec_data_transposed = rec_data_transposed.apply(pd.to_numeric, errors='coerce')

# Display the transformed data
print(rec_data_transposed.head(8))

# Prepare data for fitting
years = rec_data_transposed.index.year
predictions_years = np.array([2030, 2040])

# Dictionary to store models and predictions
models_predictions = {}

# Fit models for each country
for country in rec_data_transposed.columns:
    # Extract country data
    y = rec_data_transposed[country].dropna()
    
    # Fit a polynomial model (degree 2 chosen for simplicity)
    model_coeffs, cov_matrix = fit_polynomial(years, y, degree=2)
    
    # Predict future values
    future_predictions = predict_polynomial(model_coeffs, predictions_years)
    
    # Calculate confidence intervals
    lower_bounds, upper_bounds = confidence_intervals(model_coeffs, cov_matrix, predictions_years)
    
    # Store the results
    models_predictions[country] = {
        "model": model_coeffs,
        "predictions": future_predictions,
        "confidence_intervals": {
            "lower": lower_bounds,
            "upper": upper_bounds
        }
    }

def fit_model(x_vals, y_vals, country_name, degree=3, prediction_start_year=1990, prediction_end_year=2040):
    # Fit a high-degree polynomial to the historical data
    high_degree = len(x_vals) - 1
    high_degree_poly_coeffs = np.polyfit(x_vals, y_vals, high_degree)
    x_high_degree_trend = np.linspace(min(x_vals), max(x_vals), 500)
    y_high_degree_trend = np.polyval(high_degree_poly_coeffs, x_high_degree_trend)

    # Fit a polynomial model for the prediction range
    trend_poly_coeffs = np.polyfit(x_vals, y_vals, degree)
    x_prediction = np.linspace(prediction_start_year, prediction_end_year, 500)
    y_prediction = np.polyval(trend_poly_coeffs, x_prediction)

    # Fit the OLS model for confidence intervals
    model = sm.OLS(y_vals, sm.add_constant(x_vals))
    fitted_model = model.fit()
    prediction = fitted_model.get_prediction(sm.add_constant(x_prediction))
    ci_lower, ci_upper = prediction.conf_int().T

    # Plot
    plt.figure(figsize=(12, 5))
    plt.scatter(x_vals, y_vals, color='white', label=f'Historical Data ({country_name})')
    plt.plot(x_high_degree_trend, y_high_degree_trend, label=f'Renewable Energy Consumption ({country_name})', color='green', linewidth=2)
    plt.plot(x_prediction, y_prediction, label='Forecast', color='orange', linewidth=2)
    plt.fill_between(x_prediction, ci_lower, ci_upper, color='yellow', alpha=0.2, label='Confidence Range')
    plt.legend(loc='best')
    plt.title(f"Renewable Energy Consumption Prediction for {country_name}")
    plt.xlabel('Year')
    plt.ylabel('% Total Renewable Energy')
    plt.show()

for country in rec_data_transposed.columns:
    data_points = rec_data_transposed[country].dropna()
    X_values = data_points.index.year.astype("int")
    y_values = data_points.values.astype("float64")
    fit_model(X_values, y_values, country, degree=3, prediction_start_year=1990, prediction_end_year=2040)

