# Pathfinder AI – Smart Career Guidance & Recommendation System
PathFinder AI is a machine-learning powered career recommendation platform that helps students and individuals identify suitable career paths. The system analyzes user interests, skills, and preferences using a trained KNN (K-Nearest Neighbors) model and produces personalized career suggestions along with a structured roadmap.

# Key Features
● **User Data Collection:** Collects users’ interests, skills, strengths, and educational background through a structured multi-step form.

● **Machine Learning Model (KNN):** Uses a trained KNN algorithm to analyze user inputs and identify the closest matching career domains.

● **Personalized Career Recommendations:** Provides tailored career suggestions based on the user’s skills, preferences, and similarity with known career profiles.

● **Career Roadmap Generation:** Offers a step-by-step roadmap including required skills, certifications, learning paths, and entry-level roles for the predicted career.

● **Interactive Web Interface:** Features a modern, responsive UI built with HTML, CSS, JavaScript, and Bootstrap for smooth user interaction.

● **Flask Backend Integration:** Connects the frontend with the machine learning model to process data and deliver predictions in real time.

● **Scalable Design:** Allows easy expansion of skills, career categories, and additional datasets for future enhancements.

● **Fast Real-Time Processing:** Generates accurate career recommendations quickly using optimized preprocessing and model integration.

# Tech Stack

● **Machine Learning:** Python, scikit-learn (KNN), NumPy, Pandas, Joblib

● **Backend:** Flask (API & routing), Python

● **Frontend:** HTML5, CSS3, JavaScript, Bootstrap

● **Model Handling:** KNN Model, StandardScaler (preprocessing)

● **Web Integration**: Flask Templates (Jinja2), Multi-step form interface

● **Database:** MySQL (planned for future Integration)

● **Deployment:** Localhost using Flask development server

# Folder Structure 

`PathFinder/
├── app.py                        
├── models/
│   ├── career_model.pkl          
│   └── scaler.pkl                
├── templates/
│   ├── index.html                
│   ├── form.html                 
│   └── results.html              
├── assets/
│   ├── css/                  
│   ├── js/                   
│   ├── vendor/               
│   └── img/                 
├── dataset/
│   └── career_dataset.csv        
├── requirements.txt              
└── README.md    `
                 
