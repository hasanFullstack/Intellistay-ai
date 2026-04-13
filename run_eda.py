import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(csv_path='pricing_dataset.csv'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run generate_dataset.py first.")
        return
        
    print(f"Loading {csv_path} for Exploratory Data Analysis...")
    df = pd.read_csv(csv_path)
    
    # Create output directory for plots
    os.makedirs('eda_plots', exist_ok=True)
    
    print("\n1. Dataset Info:")
    print("-" * 30)
    print(df.info())
    
    print("\n2. Statistical Summary:")
    print("-" * 30)
    print(df.describe().round(2))
    
    # Set plotting style
    plt.style.use('default')
    # Suppress warnings for seaborn style
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\nGenerating visual plots into 'eda_plots/' directory...")
    
    # Plot 1: Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix (What drives the optimal price?)')
    plt.tight_layout()
    plt.savefig('eda_plots/1_correlation_matrix.png')
    plt.close()
    
    # Plot 2: Price vs Occupancy Rate
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='occupancy_rate', y='optimal_price', hue='has_ac', data=df, alpha=0.5)
    plt.title('Optimal Price vs Occupancy Rate (Blue=No AC, Orange=AC)')
    plt.savefig('eda_plots/2_price_vs_occupancy.png')
    plt.close()
    
    # Plot 3: Distribution of Optimal Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(df['optimal_price'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Optimal AI Prices')
    plt.xlabel('Suggested Price (Rs)')
    plt.savefig('eda_plots/3_price_distribution.png')
    plt.close()
    
    # Plot 4: Impact of Distance to University
    plt.figure(figsize=(10, 6))
    # Group into bins for cleaner visualization
    df['dist_bin'] = pd.cut(df['distance_to_university'], bins=[0, 2, 5, 10, 20], labels=['<2km', '2-5km', '5-10km', '>10km'])
    sns.boxplot(x='dist_bin', y='optimal_price', data=df)
    plt.title('Price Variation by Distance to University')
    plt.savefig('eda_plots/4_price_by_distance.png')
    plt.close()
    
    print("EDA Complete! Check the 'eda_plots' folder for the generated graphs.")

if __name__ == "__main__":
    run_eda()
