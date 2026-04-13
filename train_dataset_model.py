import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_isolated_model(csv_path='pricing_dataset.csv', model_output='pricing_model_dataset.pkl'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run generate_dataset.py first.")
        return
        
    print(f"Loading generated dataset: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Prepare Features (X) and Target (y)
    # We use all features except the target variable
    X = df.drop(columns=['optimal_price', 'dist_bin'], errors='ignore')
    y = df['optimal_price']
    
    print(f"Features used for training: {list(X.columns)}")
    
    # 2. Split Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 3. Configure XGBoost Model
    # We use slightly deeper trees because we have 10,000 rows and more complex features now
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6, 
        random_state=42
    )
    
    # 4. Train the Model
    print("Training XGBoost Regressor Model...")
    model.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Model Performance ---")
    print(f"Mean Absolute Error (MAE): Rs {mae:.2f}")
    print(f"R² Score: {r2:.4f} (1.0 is perfect prediction)")
    print("--------------------------\n")
    
    # 6. Save the Model
    joblib.dump(model, model_output)
    print(f"Model successfully saved to {model_output}")
    print("(Note: This is isolated and will not overwrite the production pricing_model.pkl)")
    
    # 7. Feature Importance Analysis
    importance = model.feature_importances_
    feature_names = X.columns
    feature_importance = sorted(zip(importance, feature_names), reverse=True)
    
    print("\nFeature Importance (What the ML Model learned to look at most):")
    for val, name in feature_importance:
        print(f"- {name}: {val:.4f}")

if __name__ == "__main__":
    train_isolated_model()
