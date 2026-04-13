import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from the backend folder to get MONGO_URI
dotenv_path = os.path.join(os.path.dirname(__dirname__), 'backend', '.env')
load_dotenv(dotenv_path)

MONGO_URI = os.getenv("MONGO_URI")

def get_city_tier(location):
    if not location:
        return 2
    loc = location.lower()
    if 'islamabad' in loc or 'lahore' in loc or 'karachi' in loc:
        return 1
    return 2

def pull_data_from_db():
    if not MONGO_URI:
        print("Error: MONGO_URI not found in backend/.env")
        return None

    try:
        client = MongoClient(MONGO_URI)
        db = client.get_default_database()
        
        hostels = list(db.hostels.find())
        rooms = list(db.rooms.find())
        
        if not rooms:
            print("No rooms found in the database. Add rooms via the app first!")
            return None
            
        print(f"Found {len(hostels)} hostels and {len(rooms)} rooms in the database.")
        
        # Mapping hostels for quick lookup
        hostel_map = {str(h['_id']): h for h in hostels}
        
        # Prepare list for DataFrame
        data = []
        current_month = datetime.datetime.now().month
        
        for room in rooms:
            hostel = hostel_map.get(str(room.get('hostelId')))
            if not hostel:
                continue
                
            total_beds = room.get('totalBeds', 1)
            available_beds = room.get('availableBeds', total_beds)
            occupancy_rate = (total_beds - available_beds) / max(total_beds, 1)
            
            base_price = room.get('pricePerBed', 10000)
            
            amenities = hostel.get('amenities', [])
            amenity_score = len(amenities) if amenities else 0
            
            city_tier = get_city_tier(hostel.get('location', ''))
            
            # Since true "optimal historical price" data doesn't exist yet in a new app,
            # we simulate the optimal price target based on actual real-world dynamic pricing rules
            # applied to the user's REAL form data.
            multiplier = 1.0
            
            # 1. Occupancy Rules (Demand)
            if occupancy_rate > 0.90:
                multiplier += 0.15
            elif occupancy_rate > 0.75:
                multiplier += 0.08
            elif occupancy_rate < 0.40:
                multiplier -= 0.12
                
            # 2. Seasonality (Peak Demand)
            if current_month in [8, 9, 1]: # Aug/Sep/Jan Admissions
                multiplier += 0.05
                
            # 3. Amenity Value (+1.5% premium per amenity offered)
            multiplier += (amenity_score * 0.015)
            
            # 4. City Tier Economics
            if city_tier == 1: # Major City
                multiplier += 0.05
            elif city_tier == 3: # Small City
                multiplier -= 0.05
                
            # 5. Room Privacy Premium
            if total_beds == 1:
                multiplier += 0.10 # Single room premium
            elif total_beds >= 4:
                multiplier -= 0.05 # Volume discount for large dorms
                
            optimal_price = round((base_price * multiplier) / 100) * 100
            
            data.append({
                'occupancy_rate': occupancy_rate,
                'total_beds': total_beds,
                'base_price': base_price,
                'month': current_month,
                'amenity_score': amenity_score,
                'city_tier': city_tier,
                'optimal_price': optimal_price
            })
            
        df = pd.DataFrame(data)
        
        # Machine learning models like XGBoost need hundreds of rows to discover patterns.
        # If the user only filled out 1 or 2 rooms via the form, we will bootstrap/augment 
        # the dataset using their actual DB room profiles as a base template.
        if len(df) < 500:
            print(f"Bootstrapping training data from your {len(df)} real DB rooms to create a robust model...")
            augmented_data = []
            for _ in range(1000): # Create 1000 variations of user's real rooms
                # Pick a random real room from the DB
                sample = df.sample(1).iloc[0]
                
                # Add slight noise to simulate different times/occupancies of this room
                noisy_occ = np.clip(sample['occupancy_rate'] + np.random.normal(0, 0.2), 0.0, 1.0)
                noisy_month = np.random.randint(1, 13)
                
                # Recalculate optimal price target for the augmented state
                noisy_mult = 1.0
                
                # Re-apply full Intrinsic Value logic to the augmented clone
                if noisy_occ > 0.90: noisy_mult += 0.15
                elif noisy_occ > 0.75: noisy_mult += 0.08
                elif noisy_occ < 0.40: noisy_mult -= 0.12
                
                if noisy_month in [8, 9, 1]: noisy_mult += 0.05
                
                noisy_mult += (sample['amenity_score'] * 0.015)
                
                tier = sample['city_tier']
                if tier == 1: noisy_mult += 0.05
                elif tier == 3: noisy_mult -= 0.05
                
                beds = sample['total_beds']
                if beds == 1: noisy_mult += 0.10
                elif beds >= 4: noisy_mult -= 0.05
                
                noisy_opt_price = round((sample['base_price'] * noisy_mult) / 100) * 100
                
                augmented_data.append({
                    'occupancy_rate': noisy_occ,
                    'total_beds': sample['total_beds'],
                    'base_price': sample['base_price'],
                    'month': noisy_month,
                    'amenity_score': sample['amenity_score'],
                    'city_tier': sample['city_tier'],
                    'optimal_price': noisy_opt_price
                })
            df = pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)
            
        return df
        
    except Exception as e:
        print(f"Database Error: {e}")
        return None

def train_model():
    print("Fetching actual forms data from MongoDB Database...")
    df = pull_data_from_db()
    
    if df is None or len(df) == 0:
        print("Training aborted. No data available.")
        return
        
    print(f"Training XGBoost on {len(df)} rows derived from your DB forms...")
    
    X = df[['occupancy_rate', 'total_beds', 'base_price', 'month', 'amenity_score', 'city_tier']]
    y = df['optimal_price']
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X, y)
    
    joblib.dump(model, 'pricing_model.pkl')
    print("Model successfully trained on actual database data and saved to pricing_model.pkl")

if __name__ == "__main__":
    train_model()
