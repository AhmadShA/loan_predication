import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()

def exclude_columns(df):
    """Exclude specific columns from the analysis"""
    return df.drop(['Col_num', 'Loan_ID'], axis=1, errors='ignore')

# 1. Overview of the Data
def overview_data(df):
    df = exclude_columns(df)
    print("\n=== Data Overview ===")
    print("\nDataset Shape:", df.shape)
    print("\nColumns and their data types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSample Data:")
    print(df.head())

# 2. Descriptive Statistics
def descriptive_stats(df):
    df = exclude_columns(df)
    print("\n=== Descriptive Statistics ===")
    print("\nNumerical Features Summary:")
    print(df.describe())
    
    # Check for skewness in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("\nSkewness in Numerical Features:")
    for col in numerical_cols:
        skewness = df[col].skew()
        print(f"{col}: {skewness:.2f}")

# 3. Visual Exploration
def numerical_visualizations(df):
    df = exclude_columns(df)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Histograms
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.close()
    
    # Boxplots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(data=df, y=col)
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig('boxplots.png')
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def categorical_visualizations(df):
    df = exclude_columns(df)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        # Bar plots
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'countplot_{col}.png')
        plt.close()

# 5. Detecting Anomalies
def detect_anomalies(df):
    df = exclude_columns(df)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    print("\n=== Anomaly Detection ===")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        print(f"\n{col}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Outlier range: [{lower_bound:.2f}, {upper_bound:.2f}]")

# 6. Feature Relationships
def analyze_feature_relationships(df):
    df = exclude_columns(df)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Scatter plots for highly correlated features
    correlation_matrix = df[numerical_cols].corr()
    high_corr_pairs = []
    
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            if abs(correlation_matrix.iloc[i,j]) > 0.5:
                high_corr_pairs.append((numerical_cols[i], numerical_cols[j]))
    
    for pair in high_corr_pairs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=pair[0], y=pair[1])
        plt.title(f'Relationship between {pair[0]} and {pair[1]}')
        plt.tight_layout()
        plt.savefig(f'scatter_{pair[0]}_{pair[1]}.png')
        plt.close()

# 7. Feature Engineering Ideas
def suggest_feature_engineering(df):
    df = exclude_columns(df)
    print("\n=== Feature Engineering Suggestions ===")
    
    # Check for potential ratio features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            print(f"Consider creating ratio: {numerical_cols[i]} / {numerical_cols[j]}")
    
    # Check for potential interaction features
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            print(f"Consider creating interaction: {numerical_cols[i]} * {numerical_cols[j]}")

# 8. Data Quality Check
def check_data_quality(df):
    df = exclude_columns(df)
    print("\n=== Data Quality Check ===")
    
    # Check for constant features
    constant_features = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_features.append(col)
    
    if constant_features:
        print("\nConstant Features:", constant_features)
    
    # Check for high cardinality
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality = []
    for col in categorical_cols:
        if df[col].nunique() > 20:  # threshold for high cardinality
            high_cardinality.append((col, df[col].nunique()))
    
    if high_cardinality:
        print("\nHigh Cardinality Features:")
        for col, unique_count in high_cardinality:
            print(f"{col}: {unique_count} unique values")

def main():
    # Load your dataset here
    df = pd.read_csv('loan_manager\ml\loan_prediction.csv')
    
    # Run all analysis functions
    overview_data(df)
    descriptive_stats(df)
    numerical_visualizations(df)
    categorical_visualizations(df)
    detect_anomalies(df)
    analyze_feature_relationships(df)
    suggest_feature_engineering(df)
    check_data_quality(df)

if __name__ == "__main__":
    main() 