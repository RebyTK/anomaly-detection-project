import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EnhancedBehaviorDataGenerator:
    def __init__(self, n_samples=5000, n_users=50):
        self.n_samples = n_samples
        self.n_users = n_users
        
    def generate_realistic_patterns(self):
        """Generate more realistic behavioral patterns with user consistency"""
        data = []
        
        for user_id in range(self.n_users):
            # Each user has consistent base patterns
            user_typing_speed_base = np.random.normal(250, 50)
            user_pressure_base = np.random.normal(0.5, 0.1)
            user_swipe_length_base = np.random.normal(150, 40)
            
            # User's preferred usage hours (circadian rhythm)
            preferred_hours = random.choice([
                [8, 9, 10, 18, 19, 20],  # Morning/evening user
                [12, 13, 14, 15, 16],    # Afternoon user
                [20, 21, 22, 23]         # Night owl
            ])
            
            samples_per_user = self.n_samples // self.n_users
            
            for session in range(samples_per_user):
                # Generate timestamp with user preference
                if random.random() < 0.8:  # 80% preferred time
                    hour = random.choice(preferred_hours)
                else:
                    hour = random.randint(0, 23)
                
                timestamp = datetime.now() - timedelta(
                    days=random.randint(0, 30),
                    hours=hour,
                    minutes=random.randint(0, 59)
                )
                
                # Typing behavior with fatigue and time-of-day effects
                time_fatigue = 1.2 if hour in [1, 2, 3, 4, 5] else 1.0
                typing_speed = user_typing_speed_base * time_fatigue + np.random.normal(0, 20)
                typing_error_rate = max(0, np.random.normal(0.03, 0.02) * time_fatigue)
                
                # App usage patterns
                if hour in [12, 13, 18, 19, 20]:  # Peak usage hours
                    app_usage_time = np.random.normal(450, 120)
                else:
                    app_usage_time = np.random.normal(180, 60)
                
                # Touch pressure with consistency
                touch_pressure = max(0.1, min(1.0, user_pressure_base + np.random.normal(0, 0.05)))
                
                # Swipe behavior
                swipe_length = user_swipe_length_base + np.random.normal(0, 25)
                swipe_angle = np.random.uniform(-np.pi, np.pi)
                
                # Movement patterns (accelerometer variance)
                if hour in [7, 8, 17, 18]:  # Commute hours
                    accel_variance = np.random.normal(0.4, 0.1)  # Higher movement
                else:
                    accel_variance = np.random.normal(0.2, 0.05)
                
                # Location patterns (work/home consistency)
                if 9 <= hour <= 17 and timestamp.weekday() < 5:  # Work hours
                    location_pattern = random.choices([0, 1], weights=[0.95, 0.05])[0]
                else:
                    location_pattern = random.choices([0, 1], weights=[0.98, 0.02])[0]
                
                # Session duration and interaction frequency
                session_duration = max(60, app_usage_time + np.random.normal(0, 30))
                interactions_per_minute = np.random.normal(15, 5)
                
                # Scroll behavior
                scroll_velocity = np.random.normal(200, 50)
                scroll_frequency = np.random.poisson(25)
                
                # Battery and device state effects
                battery_level = random.uniform(0.1, 1.0)
                device_temperature = np.random.normal(35, 5)  # Celsius
                
                # Performance metrics
                app_response_time = np.random.normal(150, 30)  # ms
                network_latency = np.random.normal(100, 40)   # ms
                
                data.append({
                    'user_id': user_id,
                    'timestamp': timestamp,
                    'hour': hour,
                    'day_of_week': timestamp.weekday(),
                    'typing_speed': typing_speed,
                    'typing_error_rate': typing_error_rate,
                    'app_usage_time': app_usage_time,
                    'touch_pressure': touch_pressure,
                    'swipe_length': swipe_length,
                    'swipe_angle': swipe_angle,
                    'accel_variance': accel_variance,
                    'location_pattern': location_pattern,
                    'session_duration': session_duration,
                    'interactions_per_minute': interactions_per_minute,
                    'scroll_velocity': scroll_velocity,
                    'scroll_frequency': scroll_frequency,
                    'battery_level': battery_level,
                    'device_temperature': device_temperature,
                    'app_response_time': app_response_time,
                    'network_latency': network_latency,
                    'night_usage_flag': 1 if 1 <= hour <= 5 else 0
                })
        
        df = pd.DataFrame(data)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Generate realistic anomalies
        df = self.generate_realistic_anomalies(df)
        
        return df
    
    def add_temporal_features(self, df):
        """Add temporal pattern features"""
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Rolling averages (user's typical behavior)
        df['typing_speed_ma7'] = df.groupby('user_id')['typing_speed'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['touch_pressure_ma7'] = df.groupby('user_id')['touch_pressure'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Deviation from personal baseline
        df['typing_speed_deviation'] = abs(df['typing_speed'] - df['typing_speed_ma7'])
        df['touch_pressure_deviation'] = abs(df['touch_pressure'] - df['touch_pressure_ma7'])
        
        # Time since last session
        df['time_since_last'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['time_since_last'] = df['time_since_last'].fillna(24)  # Assume 24h for first session
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def generate_realistic_anomalies(self, df, anomaly_rate=0.03):
        """Generate realistic anomalies based on behavioral understanding"""
        n_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = random.sample(range(len(df)), n_anomalies)
        
        df['label'] = 0  # Normal by default
        
        for idx in anomaly_indices:
            df.loc[idx, 'label'] = 1
            
            # Choose anomaly type
            anomaly_type = random.choice([
                'account_takeover', 'bot_behavior', 'device_change', 
                'location_spoofing', 'stress_typing', 'impaired_usage'
            ])
            
            if anomaly_type == 'account_takeover':
                # Sudden change in all behavioral patterns
                df.loc[idx, 'typing_speed'] *= random.uniform(0.3, 0.7)
                df.loc[idx, 'touch_pressure'] *= random.uniform(1.5, 2.5)
                df.loc[idx, 'swipe_length'] *= random.uniform(0.4, 0.8)
                
            elif anomaly_type == 'bot_behavior':
                # Too consistent/mechanical behavior
                df.loc[idx, 'typing_speed'] = df.loc[idx, 'typing_speed_ma7']  # Exact average
                df.loc[idx, 'interactions_per_minute'] *= random.uniform(3, 8)
                df.loc[idx, 'app_response_time'] *= 0.1  # Inhumanly fast
                
            elif anomaly_type == 'device_change':
                # Different device characteristics
                df.loc[idx, 'touch_pressure'] *= random.uniform(2, 4)
                df.loc[idx, 'device_temperature'] += random.uniform(15, 25)
                df.loc[idx, 'accel_variance'] *= random.uniform(0.2, 0.6)
                
            elif anomaly_type == 'location_spoofing':
                df.loc[idx, 'location_pattern'] = 1
                df.loc[idx, 'network_latency'] *= random.uniform(3, 8)
                
            elif anomaly_type == 'stress_typing':
                # Erratic typing under stress
                df.loc[idx, 'typing_error_rate'] *= random.uniform(5, 15)
                df.loc[idx, 'typing_speed'] *= random.uniform(0.6, 1.8)
                
            elif anomaly_type == 'impaired_usage':
                # Impaired (tired/intoxicated) usage patterns
                df.loc[idx, 'typing_error_rate'] *= random.uniform(3, 10)
                df.loc[idx, 'swipe_length'] *= random.uniform(0.3, 0.7)
                df.loc[idx, 'scroll_velocity'] *= random.uniform(0.2, 0.6)
        
        return df

# Generate enhanced dataset
generator = EnhancedBehaviorDataGenerator(n_samples=10000, n_users=100)
enhanced_data = generator.generate_realistic_patterns()

# Save dataset
enhanced_data.to_csv("enhanced_behavior_dataset.csv", index=False)

print("✅ Enhanced dataset generated!")
print(f"Total samples: {len(enhanced_data)}")
print(f"Total users: {enhanced_data['user_id'].nunique()}")
print(f"Anomaly rate: {enhanced_data['label'].mean():.3f}")

# Display sample statistics
print("\nFeature Statistics:")
feature_cols = ['typing_speed', 'typing_error_rate', 'touch_pressure', 
                'session_duration', 'interactions_per_minute']
print(enhanced_data[feature_cols].describe())