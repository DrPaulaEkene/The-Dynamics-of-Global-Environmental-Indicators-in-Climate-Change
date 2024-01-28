# imports and function definitions


import numpy as np  # package for arrays
import pandas as pd  # package for analysis
import seaborn as sns  # package for visualization
import matplotlib.pyplot as plt  # package for visualization
from sklearn.cluster import KMeans  # Importing Kmean from
from sklearn.metrics import silhouette_score   # to calculate Silhouete score
from sklearn.preprocessing import StandardScaler  # for data normalization
from scipy.optimize import curve_fit  # For fitting
import statsmodels.api as sm  # for statistical analysis
from errors import error_prop  # import error.py


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


def truncate_dataset(df, number_of_rows):
    """Prune the dataset to a specified number of rows
    """
    return df.head(number_of_rows)


def rename_dataframe(df, new_name):
    """ Rename the dataframe
    """
    df.name = new_name
    return df


def process_data(df):
    """ Sort the climate data to split it into separate
    sets for each indicator.

    Args:
        df(DataFrame): The original DataFrame.

    Returns:
        dict: A dictionary containing two versions of DataFrames for each
        indicator
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
        df_countries_as_columns = indicator_df.set_index(
            'Country Name').transpose()

        # Store in dictionary
        dataframes_by_indicator[indicator] = {
            'years_as_columns': df_years_as_columns,
            'countries_as_columns': df_countries_as_columns
        }

    return distinct_indicators, dataframes_by_indicator


def kmeans_clustering(data, indicators, n_clusters=3):
    """ Perform K-Means clustering on the given data and create visualizations
    Args:
        data(DataFrame): The DataFrame containing the indicators data.
    indicators(list): List of indicator names.
    n_clusters(int): Number of clusters for K-Means.

    Returns:
        DataFrame: The DataFrame with cluster labels.
    """

    # Normalizing the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[indicators])

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_data)
    data['Cluster'] = kmeans.labels_
    return data


def fit_polynomial(x, y, degree=2):
    """ Fits a polynomial of a given degree to the data.
    Returns the fitted polynomial coefficients and the covariance matrix.
    """
    # Set initial guess for the polynomial coefficients to ones
    initial_guess = np.ones(degree + 1)
    coeffs, cov_matrix = curve_fit(lambda t,
                                   *coeffs: np.polyval(coeffs, t),
                                   x, y, p0=initial_guess)
    return coeffs, cov_matrix


def predict_polynomial(model, years):
    """ Uses the fitted polynomial model to predict values for given years.
    """
    return np.polyval(model, years)


# Read excel file for Climate Data
file = 'Climate_Data_Extract_1990_2020_5Yr_expanded.xlsx'
df = load_dataset(file)

# Clean the dataset
df = format_headers(df)
df = replace_missing_values(df, '..', 0)
df = convert_to_numeric(df)
df = truncate_dataset(df, 400)

# Rename data frame
climate_data = rename_dataframe(df, 'climate_data')

# Process the climate_data to split by indicators
distinct_indicators, split_dataframes = process_data(climate_data)

# Process climate data to split by indicators
selected_indicators = [
    'CO2 emissions (metric tons per capita)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Urban population (% of total population)',
    'Forest area (% of land area)'
]

# Filter and prepare data for each selected indicator
trend_data = {}
for indicator in selected_indicators:
    indicator_df = df[df['Series Name'] == indicator].drop(
        ['Series Name', 'Series Code', 'Country Code'], axis=1)
    indicator_df.set_index('Country Name', inplace=True)
    trend_data[indicator] = indicator_df

# Merge the dataframes of the selected indicators for correlation analysis
merged_df = pd.DataFrame()
for indicator in selected_indicators:

    # Take the mean across years for each country for each indicator
    merged_df[indicator] = trend_data[indicator].mean(axis=1)


# Calculate the correlation matrix
correlation_matrix = merged_df.corr()


# Shorten indicator names for better visualization
name_mapping = {
    'Renewable energy consumption (% of total final energy consumption)':
        'Renewable Energy',
    'Urban population (% of total population)': 'Urban Population',
    'Forest area (% of land area)': 'Forest area',
    'CO2 emissions (metric tons per capita)': 'Co2 emissions'
}
merged_df.rename(columns=name_mapping, inplace=True)


# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(merged_df.corr(), annot=True, cmap='plasma', fmt=".2f",
            linewidths=.5, annot_kws={"size": 13})
plt.title("Correlation Matrix of Climate Indicators")
plt.show()


# Select the two weakly correlated indicators for clustering
indicator1 = 'Renewable Energy'
indicator2 = 'Urban Population'


# Prepare the data for clustering, drop non values
cluster_data = merged_df[[indicator1, indicator2]].dropna()


# Run KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data['Cluster'] = kmeans.fit_predict(cluster_data)
cluster_labels = kmeans.fit_predict(cluster_data)


# Plot clusters
plt.figure(figsize=(12, 8))
cluster_names = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'}
colors = ['blue', 'orangered', 'green']

# Scatter plot for data points with different colors for each cluster
for cluster_label, color in zip(cluster_names.keys(), colors):
    cluster_data_subset = cluster_data[cluster_data['Cluster'] ==
                                       cluster_label]
    plt.scatter(cluster_data_subset[indicator1],
                cluster_data_subset[indicator2],
                color=color, marker='o', s=120,
                label=f"{cluster_names[cluster_label]}")

# Extract cluster centers
cluster_centers = kmeans.cluster_centers_

# Scatter plot for cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='gold', s=200,
            alpha=0.9, marker='*', label='Cluster centers')

# Add titles and labels, grids, legends
for i, center in enumerate(cluster_centers):
    plt.text(center[0], center[1], f'Center {i}',
             horizontalalignment='center', verticalalignment='center',
             color='black', fontsize=15)
plt.title(f"Cluster Analysis: {indicator1} vs. {indicator2}", fontsize=16)
plt.xlabel('Renewable (% of total final) energy consumption', fontsize=14)
plt.ylabel('Urban population (% of total population)', fontsize=14)
plt.grid(True)
plt.legend()

# Show plot
plt.show()

# Calculate the silhouette score
silhouette_avg = silhouette_score(cluster_data[[indicator1, indicator2]],
                                  cluster_labels)

print(f"The average silhouette score for 3 clusters is: {silhouette_avg}")


# Compute the aggregation function for the cluster summary table
cluster_summary = cluster_data.groupby('Cluster').agg(
    Number_of_Countries=('Cluster', 'count'),
    Average_Renewable_Energy_Consumption=(indicator1, 'mean'),
    Average_Urban_Population=(indicator2, 'mean')
).reset_index()

print(cluster_summary.head(6))


# identify the countries for each cluster
for cluster_number in range(3):  # Since we have 3 clusters (0, 1, 2)
    countries = cluster_data[cluster_data['Cluster']
                             == cluster_number].index.tolist()
    print(f"Countries in Cluster {cluster_number}: {countries}")


# Extract fitting data for the specified countries, one from each cluster
countries = ["Germany", "China", "Nigeria"]
filtered_data = climate_data[climate_data['Country Name'].isin(countries)]
rec_data = filtered_data[filtered_data['Series Name'].str.contains(
    "Renewable energy consumption")]
rec_data_cleaned = rec_data.drop(columns=["Series Name", "Series Code",
                                          "Country Code"])
rec_data_cleaned.set_index("Country Name", inplace=True)


# Transposing and converting years
rec_data_transposed = rec_data_cleaned.T
rec_data_transposed.index = pd.to_datetime(
    rec_data_transposed.index.str.extract("(\d{4})")[0])
rec_data_transposed = rec_data_transposed.apply(pd.to_numeric, errors='coerce')
rec_data_transposed.head(8)


# Preparing prediction and forecasting data
years = rec_data_transposed.index.year
selected_countries = ["Germany", "China", "Nigeria"]
selected_countries_predictions = {}


# Fitting models and making predictions for each selected country
for country in selected_countries:
    if country in rec_data_transposed.columns:
        data_points = rec_data_transposed[country].dropna()

        # LOWESS smoothing
        lowess = sm.nonparametric.lowess
        X_values = data_points.index.year.astype("int")
        y_values = data_points.values.astype("float64")
        y_smoothed = lowess(y_values, X_values, frac=0.1)
        x_smoothed, y_smoothed = y_smoothed[:, 0], y_smoothed[:, 1]

        # Fit polynomial model
        trend_poly_coeffs, cov_matrix = fit_polynomial(X_values, y_values,
                                                       degree=2)

        # Generate predictions
        predictions_years = np.array([2030, 2040])
        x_prediction = np.linspace(min(X_values), max(predictions_years), 500)
        y_prediction = np.polyval(trend_poly_coeffs, x_prediction)

        # Calculate confidence intervals
        sigma = error_prop(x_prediction,
                           lambda x, *params: np.polyval(params, x),
                           trend_poly_coeffs, cov_matrix)
        ci_lower = y_prediction - sigma
        ci_upper = y_prediction + sigma

        # Visualization
        plt.figure(figsize=(12, 8))
        plt.plot(x_prediction, y_prediction, label='Forecast',
                 color='blue', linewidth=4)
        plt.plot(x_smoothed, y_smoothed,
                 label='Historical Data', color='red',
                 linewidth=4, linestyle='--')
        plt.fill_between(x_prediction, ci_lower, ci_upper,
                         color='gold', alpha=0.2,
                         label='Confidence Range')
        plt.legend(loc='best')
        plt.title(f"Renewable Energy Consumption Prediction for {country}",
                  fontsize=18)
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('% Total Renewable Energy', fontsize=16)
        plt.show()
