import pandas as pd
import numpy as np
import os

def generate_dataset(num_samples=10000):
    print(f"Generating {num_samples} rows of synthetic hostel pricing data...")
    
    # 1. Base Variables
    # Normal distribution of base prices (Rs 5k to 25k)
    base_price = np.random.normal(12000, 3000, num_samples)
    base_price = np.clip(base_price, 5000, 25000)
    base_price = np.round(base_price / 500) * 500 # Round to nearest 500
    
    total_beds = np.random.choice([1, 2, 4, 6, 8], num_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    occupancy_rate = np.random.beta(a=5, b=2, size=num_samples) # Skewed towards higher occupancy
    month = np.random.randint(1, 13, num_samples)
    city_tier = np.random.choice([1, 2, 3], num_samples, p=[0.4, 0.4, 0.2])
    
    # 2. Extra Features (The new ones requested for this isolated pipeline)
    distance_to_university = np.random.exponential(scale=3.5, size=num_samples) # km
    distance_to_university = np.clip(distance_to_university, 0.1, 15.0)
    
    student_rating = np.random.normal(4.1, 0.6, num_samples)
    student_rating = np.clip(student_rating, 1.0, 5.0)
    
    has_ac = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
    has_wifi = np.random.choice([0, 1], num_samples, p=[0.1, 0.9]) # Most have wifi
    is_weekend = np.random.choice([0, 1], num_samples, p=[0.71, 0.29]) # ~2/7 days are weekend
    
    competitor_avg_price = base_price + np.random.normal(0, 1500, num_samples)
    competitor_avg_price = np.round(competitor_avg_price / 500) * 500
    
    # 3. Calculate "Ground Truth" Target Variable (optimal_price)
    # This is the hidden formula the ML model will try to learn.
    optimal_price = base_price.copy()
    
    # Demand factors
    optimal_price = np.where(occupancy_rate > 0.90, optimal_price * 1.15, optimal_price)
    optimal_price = np.where(occupancy_rate < 0.40, optimal_price * 0.85, optimal_price)
    
    # Seasonality
    peak_months = np.isin(month, [8, 9, 1])
    optimal_price = np.where(peak_months, optimal_price * 1.08, optimal_price)
    
    # New Feature impacts
    optimal_price = np.where(distance_to_university < 2.0, optimal_price * 1.10, optimal_price) # Close to uni premium
    optimal_price = np.where(distance_to_university > 7.0, optimal_price * 0.90, optimal_price) # Far away discount
    
    optimal_price = np.where(student_rating >= 4.5, optimal_price * 1.05, optimal_price)
    optimal_price = np.where(student_rating < 3.0, optimal_price * 0.80, optimal_price)
    
    optimal_price = np.where(has_ac == 1, optimal_price * 1.20, optimal_price) # AC is a major premium
    optimal_price = np.where(is_weekend == 1, optimal_price * 1.05, optimal_price)
    
    # Add some random market noise (real life isn't perfectly formulaic)
    noise = np.random.normal(0, 400, num_samples)
    optimal_price = optimal_price + noise
    optimal_price = np.round(optimal_price / 100) * 100 # Round to nearest 100 Rs
    
    # Create DataFrame
    df = pd.DataFrame({
        'base_price': base_price,
        'total_beds': total_beds,
        'occupancy_rate': occupancy_rate,
        'month': month,
        'city_tier': city_tier,
        'distance_to_university': np.round(distance_to_university, 1),
        'student_rating': np.round(student_rating, 1),
        'has_ac': has_ac,
        'has_wifi': has_wifi,
        'is_weekend': is_weekend,
        'competitor_avg_price': competitor_avg_price,
        'optimal_price': optimal_price
    })
    
    # Save to CSV
    output_path = 'pricing_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path} with {len(df)} rows and {len(df.columns)} columns.")
    
    # Show a quick preview
    print("\nDataset Preview:")
    print(df.head())

if __name__ == "__main__":
    generate_dataset()
