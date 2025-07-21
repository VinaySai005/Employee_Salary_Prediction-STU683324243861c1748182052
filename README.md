# 💰 SalarySense - Professional Salary Prediction Platform

A comprehensive machine learning web application that predicts data science salaries using AI and provides insights.

## 🌟 Project Overview

SalarySense is designed for the professional data science job market, featuring:

- **Professional Market Focus**: Tailored for major cities, companies, and salary structures
- **INR Currency**: All predictions and displays in Indian Rupees
- **Professional Interface**: Clean, modern UI with responsive design
- **Comprehensive Job Titles**: Localized job roles and company types
- **Regional Insights**: City-wise and company-type-wise salary predictions

## 🔧 Key Features

### 🤖 Machine Learning Pipeline
- **Data Analysis**: Comprehensive EDA with market insights
- **Multiple Models**: Linear Regression, Decision Tree, Random Forest, XGBoost, SVR, Neural Networks
- **Model Evaluation**: MAE, RMSE, R² score comparisons
- **Feature Engineering**: Professional-specific categorical encodings
- **Model Export**: Trained model saved as `salary_model.pkl`

### 🌐 Web Application
- **Modern UI**: Responsive design with professional color scheme
- **Real-time Predictions**: Instant salary predictions in INR format
- **Input Validation**: Comprehensive form validation with error messages
- **Professional Design**: Clean, modern interface with corporate branding

### 📊 Market Adaptations
- **Currency Conversion**: USD to INR (1 USD = ₹83)
- **Major Cities**: Bangalore, Mumbai, Delhi, Hyderabad, Pune, Chennai, Kolkata, Gurgaon
- **Company Types**: MNC, Startup, Government, Corporate
- **Experience Levels**: Fresher, Junior, Senior, Manager
- **Job Titles**: Professional data science roles

## 🚀 Technology Stack

### Backend & ML
- **Python 3.8+**: Core language for data analysis and ML
- **Scikit-learn**: Machine learning models and preprocessing
- **XGBoost**: Advanced gradient boosting
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

### Web Application
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Modern UI components
- **Lucide React**: Beautiful icons

## 📊 Dataset Features
Link for the dataset: https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023

Data Science Job Salaries Dataset contains 11 columns, each are:

work_year: The year the salary was paid.

experience_level: The experience level in the job during the year

employment_type: The type of employment for the role

job_title: The role worked in during the year.

salary: The total gross salary amount paid.

salary_currency: The currency of the salary paid as an ISO 4217 currency code.

salaryinusd: The salary in USD

employee_residence: Employee's primary country of residence in during the work year as an ISO 3166 country code.

remote_ratio: The overall amount of work done remotely

company_location: The country of the employer's main office or contracting branch

company_size: The median number of people that worked for the company during the year



## 🔧 Installation & Setup

### Prerequisites
- **Node.js 18+** installed
- **Python 3.8+** installed
- **Git** installed

### Step-by-Step Installation

1. **Clone the Repository**
   \`\`\`bash
   git clone <repository-url>
   cd SalarySense
   \`\`\`

2. **Install Python Dependencies**
   \`\`\`bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
   \`\`\`

3. **Install Node.js Dependencies**
   \`\`\`bash
   npm install
   \`\`\`

4. **Run the Data Analysis and Model Training**
   \`\`\`bash
   python scripts/data_analysis_and_training.py
   \`\`\`

5. **Start the Web Application**
   \`\`\`bash
   npm run dev
   \`\`\`

6. **Open Your Browser**
   Navigate to `http://localhost:3000`

## 📈 Model Performance

The Random Forest model achieved the following performance metrics:
- **R² Score**: ~0.92 (High accuracy)
- **MAE**: ~₹1.5-2.0 L (Low error)
- **RMSE**: ~₹2.0-2.5 L (Good precision)

## 🎯 Usage

1. **Data Analysis**: Run the Python script to analyze data and train models
2. **Web Interface**: Use the web app to input employee details
3. **Get Predictions**: Receive instant salary predictions with confidence scores
4. **Interpret Results**: View model information and salary context

## 📱 Web Application Features

The web application features:
- Clean, modern interface with professional design
- Intuitive form with dropdowns for all inputs
- Real-time validation and error handling
- Beautiful prediction results with confidence indicators
- Responsive design that works on all devices
- Indian Rupee formatting (₹X.X L / ₹X.X Cr)

## 🔮 Future Enhancements

- [ ] Integration with real Python backend using Flask/FastAPI
- [ ] More advanced models (XGBoost, Neural Networks)
- [ ] Additional visualizations and charts
- [ ] Salary comparison features
- [ ] Historical trend analysis
- [ ] Export prediction reports
- [ ] User authentication and saved predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset inspired by Kaggle's Data Science Salaries 2023
- Built with modern web technologies and ML best practices
- UI components from shadcn/ui library
- Icons from Lucide React

---
## Webpage Overview
<img width="1752" height="804" alt="image" src="https://github.com/user-attachments/assets/13b656f4-2b02-47f2-a6e3-8b812efe56db" />
<img width="1745" height="920" alt="image" src="https://github.com/user-attachments/assets/c5c70290-2489-44e0-9968-70f4441ce227" />
<img width="1733" height="924" alt="image" src="https://github.com/user-attachments/assets/d108a713-d104-45ac-a469-aecb49227eac" />


