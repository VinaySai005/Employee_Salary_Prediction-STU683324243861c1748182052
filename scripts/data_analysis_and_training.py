import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("🇮🇳 SalarySense: Indian Data Science Salary Predictor")
print("=" * 60)

# Create sample Indian-focused dataset
print("📊 Creating Indian Data Science Salary Dataset...")
np.random.seed(42)

# Indian job title mapping
indian_job_titles = [
    'Data Scientist', 'Data Engineer', 'Data Analyst', 'ML Engineer',
    'Business Analyst', 'Analytics Manager', 'Data Architect', 
    'Research Scientist', 'AI Engineer', 'Product Analyst'
]

# Indian cities and company types
indian_cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata', 'Gurgaon']
company_types = ['MNC', 'Startup', 'Government', 'Indian_Corporate']
experience_levels = ['Fresher', 'Junior', 'Senior', 'Manager']

# Generate Indian-focused dataset
n_samples = 1200
data = {
    'work_year': np.random.choice([2020, 2021, 2022, 2023], n_samples),
    'experience_level': np.random.choice(experience_levels, n_samples, p=[0.25, 0.35, 0.3, 0.1]),
    'employment_type': np.random.choice(['Full_Time', 'Part_Time', 'Contract', 'Freelance'], 
                                      n_samples, p=[0.8, 0.05, 0.1, 0.05]),
    'job_title': np.random.choice(indian_job_titles, n_samples),
    'employee_location': np.random.choice(indian_cities, n_samples),
    'remote_ratio': np.random.choice([0, 50, 100], n_samples, p=[0.4, 0.35, 0.25]),
    'company_location': np.random.choice(indian_cities, n_samples),
    'company_type': np.random.choice(company_types, n_samples, p=[0.4, 0.3, 0.15, 0.15])
}

# Generate realistic Indian salary data (in INR)
base_salaries_inr = {
    'Fresher': 500000,    # 5 LPA
    'Junior': 800000,     # 8 LPA  
    'Senior': 1500000,    # 15 LPA
    'Manager': 2500000    # 25 LPA
}

job_multipliers = {
    'Data Scientist': 1.2, 'Data Engineer': 1.15, 'Data Analyst': 0.9,
    'ML Engineer': 1.25, 'Business Analyst': 0.85, 'Analytics Manager': 1.4,
    'Data Architect': 1.5, 'Research Scientist': 1.3, 'AI Engineer': 1.35,
    'Product Analyst': 1.1
}

city_multipliers = {
    'Bangalore': 1.2, 'Mumbai': 1.15, 'Delhi': 1.1, 'Hyderabad': 1.05,
    'Pune': 1.0, 'Chennai': 0.95, 'Kolkata': 0.85, 'Gurgaon': 1.1
}

company_multipliers = {
    'MNC': 1.3, 'Startup': 1.0, 'Government': 0.7, 'Indian_Corporate': 0.9
}

salaries_inr = []
for i in range(n_samples):
    base = base_salaries_inr[data['experience_level'][i]]
    job_mult = job_multipliers[data['job_title'][i]]
    city_mult = city_multipliers[data['employee_location'][i]]
    company_mult = company_multipliers[data['company_type'][i]]
    remote_mult = 1.05 if data['remote_ratio'][i] == 100 else 1.0
    year_mult = 1.0 + (data['work_year'][i] - 2020) * 0.05  # 5% growth per year
    
    salary = base * job_mult * city_mult * company_mult * remote_mult * year_mult
    salary += np.random.normal(0, salary * 0.15)  # Add realistic noise
    salaries_inr.append(max(300000, int(salary)))  # Minimum 3 LPA

data['salary_in_inr'] = salaries_inr
df = pd.DataFrame(data)

print(f"✅ Dataset created successfully! Shape: {df.shape}")
print(f"📈 Salary range: ₹{df['salary_in_inr'].min():,} - ₹{df['salary_in_inr'].max():,}")

# Dataset overview
print("\n📋 Dataset Overview:")
print(df.info())
print("\n📊 First 5 rows:")
print(df.head())

# Check for missing values
print("\n🔍 Missing Values Check:")
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("✅ No missing values found!")

# Basic salary statistics
print("\n💰 Salary Statistics (INR):")
salary_stats = df['salary_in_inr'].describe()
print(salary_stats)

def format_inr(amount):
    """Format amount in Indian currency format"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.1f} Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.1f} L"
    else:
        return f"₹{amount:,.0f}"

print(f"\n💡 Salary Range: {format_inr(salary_stats['min'])} - {format_inr(salary_stats['max'])}")
print(f"📊 Average Salary: {format_inr(salary_stats['mean'])}")
print(f"📈 Median Salary: {format_inr(salary_stats['50%'])}")

# Extensive Exploratory Data Analysis
print("\n🎨 Creating comprehensive visualizations...")

# Create comprehensive EDA plots
fig = plt.figure(figsize=(20, 24))
fig.suptitle('SalarySense: Indian Data Science Salary Analysis 🇮🇳', fontsize=20, fontweight='bold', y=0.98)

# 1. Salary Distribution
plt.subplot(4, 3, 1)
plt.hist(df['salary_in_inr']/100000, bins=30, alpha=0.7, color='#FF6B35', edgecolor='black')
plt.title('Salary Distribution (in Lakhs INR)', fontweight='bold')
plt.xlabel('Salary (Lakhs ₹)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 2. Experience Level vs Salary
plt.subplot(4, 3, 2)
exp_salary = df.groupby('experience_level')['salary_in_inr'].mean().reindex(experience_levels)
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
bars = plt.bar(exp_salary.index, exp_salary.values/100000, color=colors, alpha=0.8)
plt.title('Average Salary by Experience Level', fontweight='bold')
plt.xlabel('Experience Level')
plt.ylabel('Average Salary (Lakhs ₹)')
plt.xticks(rotation=45)
for bar, val in zip(bars, exp_salary.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val/100000:.1f}L', ha='center', fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Top Job Titles by Salary
plt.subplot(4, 3, 3)
job_salary = df.groupby('job_title')['salary_in_inr'].mean().sort_values(ascending=True)
plt.barh(job_salary.index, job_salary.values/100000, color='#00BCD4', alpha=0.8)
plt.title('Average Salary by Job Title', fontweight='bold')
plt.xlabel('Average Salary (Lakhs ₹)')
plt.grid(True, alpha=0.3)

# 4. City-wise Salary Analysis
plt.subplot(4, 3, 4)
city_salary = df.groupby('employee_location')['salary_in_inr'].mean().sort_values(ascending=False)
plt.bar(city_salary.index, city_salary.values/100000, color='#E91E63', alpha=0.8)
plt.title('Average Salary by City', fontweight='bold')
plt.xlabel('City')
plt.ylabel('Average Salary (Lakhs ₹)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 5. Company Type vs Salary
plt.subplot(4, 3, 5)
company_salary = df.groupby('company_type')['salary_in_inr'].mean()
plt.bar(company_salary.index, company_salary.values/100000, color='#795548', alpha=0.8)
plt.title('Average Salary by Company Type', fontweight='bold')
plt.xlabel('Company Type')
plt.ylabel('Average Salary (Lakhs ₹)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 6. Remote Work Impact
plt.subplot(4, 3, 6)
remote_salary = df.groupby('remote_ratio')['salary_in_inr'].mean()
plt.bar(remote_salary.index.astype(str), remote_salary.values/100000, color='#607D8B', alpha=0.8)
plt.title('Salary by Remote Work Ratio', fontweight='bold')
plt.xlabel('Remote Work %')
plt.ylabel('Average Salary (Lakhs ₹)')
plt.grid(True, alpha=0.3)

# 7. Salary Trend by Year
plt.subplot(4, 3, 7)
year_salary = df.groupby('work_year')['salary_in_inr'].mean()
plt.plot(year_salary.index, year_salary.values/100000, marker='o', linewidth=3, 
         markersize=8, color='#FF5722')
plt.title('Salary Trend Over Years', fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Average Salary (Lakhs ₹)')
plt.grid(True, alpha=0.3)

# 8. Employment Type Distribution
plt.subplot(4, 3, 8)
emp_counts = df['employment_type'].value_counts()
plt.pie(emp_counts.values, labels=emp_counts.index, autopct='%1.1f%%', 
        colors=['#4CAF50', '#FF9800', '#2196F3', '#E91E63'])
plt.title('Employment Type Distribution', fontweight='bold')

# 9. Experience Level Distribution
plt.subplot(4, 3, 9)
exp_counts = df['experience_level'].value_counts().reindex(experience_levels)
plt.pie(exp_counts.values, labels=exp_counts.index, autopct='%1.1f%%',
        colors=['#9C27B0', '#3F51B5', '#009688', '#FF5722'])
plt.title('Experience Level Distribution', fontweight='bold')

# 10. Salary Box Plot by Experience
plt.subplot(4, 3, 10)
df_plot = df.copy()
df_plot['salary_lakhs'] = df_plot['salary_in_inr'] / 100000
sns.boxplot(data=df_plot, x='experience_level', y='salary_lakhs', 
           order=experience_levels, palette='Set2')
plt.title('Salary Distribution by Experience Level', fontweight='bold')
plt.xlabel('Experience Level')
plt.ylabel('Salary (Lakhs ₹)')
plt.xticks(rotation=45)

# 11. Company Type vs Remote Work
plt.subplot(4, 3, 11)
pivot_data = df.pivot_table(values='salary_in_inr', index='company_type', 
                           columns='remote_ratio', aggfunc='mean')
sns.heatmap(pivot_data/100000, annot=True, fmt='.1f', cmap='YlOrRd', 
           cbar_kws={'label': 'Salary (Lakhs ₹)'})
plt.title('Salary Heatmap: Company Type vs Remote Work', fontweight='bold')
plt.xlabel('Remote Work %')
plt.ylabel('Company Type')

# 12. Top Cities by Job Count
plt.subplot(4, 3, 12)
city_counts = df['employee_location'].value_counts()
plt.bar(city_counts.index, city_counts.values, color='#8BC34A', alpha=0.8)
plt.title('Job Opportunities by City', fontweight='bold')
plt.xlabel('City')
plt.ylabel('Number of Jobs')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Correlation Analysis
print("\n🔥 Creating correlation analysis...")
df_encoded = df.copy()

# Encode categorical variables for correlation
label_encoders = {}
categorical_cols = ['experience_level', 'employment_type', 'job_title', 
                   'employee_location', 'company_location', 'company_type']

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Create correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df_encoded.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
           square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix - SalarySense 🇮🇳', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Data Preprocessing for Machine Learning
print("\n🔧 Preprocessing data for machine learning models...")

# Prepare features and target
X = df.drop('salary_in_inr', axis=1)
y = df['salary_in_inr']

# Encode categorical variables
for col in categorical_cols:
    X[col] = label_encoders[col].transform(X[col])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Training set: {X_train_scaled.shape}")
print(f"✅ Test set: {X_test_scaled.shape}")

# Machine Learning Model Training and Evaluation
print("\n🤖 Training multiple regression models...")

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
    'Support Vector Regressor': SVR(kernel='rbf'),
    'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

results = {}
print("\n📊 Model Performance Results:")
print("-" * 80)

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train the model
    if name in ['Support Vector Regressor', 'Neural Network (MLP)']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'predictions': y_pred,
        'scaled': name in ['Support Vector Regressor', 'Neural Network (MLP)']
    }
    
    print(f"  📈 MAE: {format_inr(mae)}")
    print(f"  📈 RMSE: {format_inr(rmse)}")
    print(f"  📈 R² Score: {r2:.4f}")

# Find the best model
best_model_name = max(results.keys(), key=lambda k: results[k]['R²'])
best_model_info = results[best_model_name]

print(f"\n🏆 Best Performing Model: {best_model_name}")
print(f"  🎯 R² Score: {best_model_info['R²']:.4f}")
print(f"  💰 MAE: {format_inr(best_model_info['MAE'])}")
print(f"  📊 RMSE: {format_inr(best_model_info['RMSE'])}")

# Model Comparison Visualization
print("\n📈 Creating model comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SalarySense: Model Performance Analysis 🇮🇳', fontsize=16, fontweight='bold')

# 1. R² Score Comparison
ax1 = axes[0, 0]
model_names = list(results.keys())
r2_scores = [results[name]['R²'] for name in model_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

bars = ax1.bar(range(len(model_names)), r2_scores, color=colors, alpha=0.8)
ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance: R² Score Comparison')
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', fontweight='bold')

# 2. MAE Comparison
ax2 = axes[0, 1]
mae_scores = [results[name]['MAE']/100000 for name in model_names]  # Convert to lakhs
bars2 = ax2.bar(range(len(model_names)), mae_scores, color=colors, alpha=0.8)
ax2.set_xlabel('Models')
ax2.set_ylabel('MAE (Lakhs ₹)')
ax2.set_title('Model Performance: Mean Absolute Error')
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# 3. Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
best_predictions = best_model_info['predictions']
ax3.scatter(y_test/100000, best_predictions/100000, alpha=0.6, color='#FF6B35', s=50)
ax3.plot([y_test.min()/100000, y_test.max()/100000], 
         [y_test.min()/100000, y_test.max()/100000], 'r--', lw=2)
ax3.set_xlabel('Actual Salary (Lakhs ₹)')
ax3.set_ylabel('Predicted Salary (Lakhs ₹)')
ax3.set_title(f'Actual vs Predicted - {best_model_name}')
ax3.grid(True, alpha=0.3)

# 4. Residuals Plot (Best Model)
ax4 = axes[1, 1]
residuals = y_test - best_predictions
ax4.scatter(best_predictions/100000, residuals/100000, alpha=0.6, color='#4CAF50', s=50)
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Salary (Lakhs ₹)')
ax4.set_ylabel('Residuals (Lakhs ₹)')
ax4.set_title(f'Residuals Plot - {best_model_name}')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature Importance Analysis
if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree']:
    print(f"\n📊 Feature Importance Analysis - {best_model_name}")
    
    if hasattr(best_model_info['model'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model_info['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance)
        
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'], 
                color='#00BCD4', alpha=0.8)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {best_model_name} (SalarySense 🇮🇳)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Save the best model and preprocessing components
print(f"\n💾 Saving the best model and preprocessing components...")

# Create comprehensive model package
model_package = {
    'model': best_model_info['model'],
    'scaler': scaler if best_model_info['scaled'] else None,
    'label_encoders': label_encoders,
    'feature_columns': list(X.columns),
    'model_name': best_model_name,
    'performance_metrics': {
        'MAE': best_model_info['MAE'],
        'RMSE': best_model_info['RMSE'],
        'R²': best_model_info['R²']
    },
    'currency': 'INR',
    'conversion_rate': 83,  # USD to INR
    'model_version': '1.0',
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'feature_mappings': {
        'experience_levels': experience_levels,
        'job_titles': indian_job_titles,
        'cities': indian_cities,
        'company_types': company_types
    }
}

# Save the model
joblib.dump(model_package, 'salary_model.pkl')
print("✅ Model package saved as 'salary_model.pkl'")

# Model Summary
print(f"\n🎉 SalarySense Model Training Complete! 🇮🇳")
print("=" * 60)
print(f"✅ Dataset Size: {len(df):,} records")
print(f"✅ Best Model: {best_model_name}")
print(f"✅ Model Accuracy (R²): {best_model_info['R²']:.4f}")
print(f"✅ Average Error: {format_inr(best_model_info['MAE'])}")
print(f"✅ Salary Range: {format_inr(df['salary_in_inr'].min())} - {format_inr(df['salary_in_inr'].max())}")
print(f"✅ Model saved and ready for deployment!")

# Sample predictions for testing
print(f"\n🧪 Sample Predictions:")
print("-" * 40)

sample_profiles = [
    {'experience_level': 'Senior', 'job_title': 'Data Scientist', 'employee_location': 'Bangalore', 'company_type': 'MNC'},
    {'experience_level': 'Junior', 'job_title': 'Data Analyst', 'employee_location': 'Mumbai', 'company_type': 'Startup'},
    {'experience_level': 'Manager', 'job_title': 'ML Engineer', 'employee_location': 'Delhi', 'company_type': 'MNC'}
]

for i, profile in enumerate(sample_profiles, 1):
    # Create sample input
    sample_input = pd.DataFrame([{
        'work_year': 2023,
        'experience_level': label_encoders['experience_level'].transform([profile['experience_level']])[0],
        'employment_type': label_encoders['employment_type'].transform(['Full_Time'])[0],
        'job_title': label_encoders['job_title'].transform([profile['job_title']])[0],
        'employee_location': label_encoders['employee_location'].transform([profile['employee_location']])[0],
        'remote_ratio': 50,
        'company_location': label_encoders['company_location'].transform([profile['employee_location']])[0],
        'company_type': label_encoders['company_type'].transform([profile['company_type']])[0]
    }])
    
    # Make prediction
    if best_model_info['scaled']:
        prediction = best_model_info['model'].predict(scaler.transform(sample_input))[0]
    else:
        prediction = best_model_info['model'].predict(sample_input)[0]
    
    print(f"{i}. {profile['experience_level']} {profile['job_title']} in {profile['employee_location']} ({profile['company_type']})")
    print(f"   Predicted Salary: {format_inr(prediction)}")

print(f"\n🚀 Ready to build the SalarySense web application!")
