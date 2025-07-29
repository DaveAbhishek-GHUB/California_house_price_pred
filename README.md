# California House Price Prediction - ML Web App

This project is a machine learning–powered web application that predicts California housing prices based on user input like location, income, and property details. It uses a trained Random Forest model and is deployed online using **Flask + Render**.

---

## Features

- Clean, user-friendly web form (Flask + HTML + CSS)
- Predicts house prices using trained ML model
- Accurate results (~89.9% R² Score)
- Professional VS Code–friendly folder structure
- Deployed online using Render

---

## Model Overview

- **Algorithm**: Random Forest Regressor
- **Performance**:  
  - **RMSE**: ~30,168  
  - **R² Score**: ~89.9%
- **Dataset**: California Housing Prices (Kaggle)

---

## Folder Structure

```

California-Housing-Price-Pred/
├── app/
│   ├── app.py                 ← Flask app backend
│   ├── templates/
│   │   └── form.html          ← Web page form
│   └── static/
│       └── style.css          ← Custom CSS
├── config/                    ← Config files (optional)
├── data/                      ← Dataset CSV (housing.csv)
├── notebooks/                 ← Jupyter notebooks for EDA, testing
├── saved\_models/
│   └── model.pkl              ← Trained ML model
├── src/                       ← Source code modules
├── main.py                    ← Main ML script (train/evaluate/save)
├── .gitignore
├── requirements.txt
├── README.md

````

---

## Local Setup

To run the project locally on your machine:

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/California-Housing-Price-Pred.git
cd California-Housing-Price-Pred
````

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Train the model (if not already trained)

```bash
python main.py
```

### 4. Run the Flask web app

```bash
python app/app.py
```

The app will be available at: [http://localhost:5000](http://localhost:5000)

---

## Deployment (Render)

App is deployed online using [Render.com](https://render.com).

### 🔧 Render Settings:

* **Build Command**:

  ```bash
  pip install -r requirements.txt
  ```

* **Start Command**:

  ```bash
  python app/app.py
  ```

---

## Sample Prediction

> **Input**:

* Longitude: -122.23
* Latitude: 37.88
* Median Income: 8.32
* Total Rooms: 880
* Total Bedrooms: 129
* Population: 322
* Ocean Proximity: INLAND

> **Predicted Price**: `$442,733.15`

---

## 🛠 Tools Used

* Python 3.10+
* Scikit-learn
* Pandas
* NumPy
* Flask
* HTML & CSS
* Render (Deployment)

---

## Created By

**Abhishek**
Built with help from AI as a real-world machine learning deployment project 💻

---

## Show Some Love!

If you like this project, consider giving the repo a ⭐ on GitHub and sharing it!

```
