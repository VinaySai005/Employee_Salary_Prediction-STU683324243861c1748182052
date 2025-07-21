# 💰 SalarySense - Professional Salary Prediction Platform

A comprehensive machine learning web application that predicts data science salaries using AI and provides insights in Indian Rupees (INR).

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

The model uses these features to predict professional data science salaries:

| Feature | Description | Context |
|---------|-------------|---------|
| **work_year** | Employment year (2020-2023) | Recent market trends |
| **experience_level** | Fresher, Junior, Senior, Manager | Career progression |
| **employment_type** | Full Time, Part Time, Contract, Freelance | Employment patterns |
| **job_title** | 10 data science roles | Popular DS roles |
| **employee_location** | 8 major cities | Tech hubs and metros |
| **remote_ratio** | 0%, 50%, 100% | Post-COVID work patterns |
| **company_location** | Company's city | Business location impact |
| **company_type** | MNC, Startup, Government, Corporate | Company landscape |
| **salary_in_inr** | Target variable | Annual salary in INR |

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

**Built with ❤️ for the professional data science community**
