## 🌾 Agri_Project 🚜
Welcome to Agri_Project — a Machine Learning-powered Crop Recommendation System for smarter agriculture!

## 📋 Project Overview
Agri_Project predicts the best crop to plant based on key parameters such as:

Nitrogen (N)

Phosphorus (P)

Potassium (K)

Temperature 🌡️

Humidity 💧

Soil pH 🧪

Rainfall 🌧️

Simply input your field data into the web app and get instant crop suggestions!

## 🛠️ Features
🖥️ Flask web interface for easy input and instant results

🤖 Machine Learning models (RandomForest, Decision Tree, SVM, Logistic Regression, Naive Bayes)

📊 Visual analytics for your dataset (heatmaps, accuracy comparisons)

⚡ Fast, user-friendly, and deployable with Docker

## 🚀 Usage Instructions
1.Clone the repository:

bash
git clone https://github.com/shlokburmi/Agri_Project.git

cd Agri_Project




2.Install dependencies:

bash
pip install -r requirements.txt

3.Prepare your data:

Ensure your crop dataset (Crop_recommendation.csv) is in the project folder.

4.Train Machine Learning Models:

bash
python train_and_save.py

5.Run the Flask App:

bash
python app.py

6.Open the Application:

Visit http://localhost:5000 to use the Crop Recommendation System.

## 📁 Repository Structure

File / Folder	Purpose
app.py	Flask backend/web server
train_and_save.py	ML training and model saving scripts
Crop_recommendation.csv	Sample crop data
models/	Saved trained model files
templates/	HTML frontend templates
requirements.txt	Python package dependencies
Dockerfile	Docker deployment file

## 🧑‍🔬 How it Works
Enter your soil and climate data in the web form

The ML model analyzes inputs and instantly suggests the best crop 🌱

Results are displayed along with any relevant insights

## ✨ Contributing
Contributions are welcome!
Feel free to open issues, fork the repo, or submit pull requests. 🛠️

## 📚 License
This project is open-source and free for academic or personal use.

Grow smarter, grow better! 🌱🌾🚜
