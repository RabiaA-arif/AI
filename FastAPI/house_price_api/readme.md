# End-to-End FastAPI House Price Prediction API

## 1. Project Overview
This project is a complete machine learning deployment example that predicts house prices in California using a trained regression model. The solution combines data science, model training, and API development into one end-to-end workflow.

The main goal is to turn a machine learning model into a real-world web service that can receive house feature values and return an estimated price in USD.

## 2. Project Objective
- Build a predictive model for house prices
- Use the California Housing dataset from scikit-learn
- Create a REST API with FastAPI
- Allow single prediction and batch prediction from CSV files
- Demonstrate practical ML deployment skills for resume and portfolio use

## 3. Business / Real-World Value
House price prediction is a strong example of supervised learning in real estate. It helps show how machine learning can be used for:
- Price estimation for buyers and sellers
- Market analysis and forecasting
- Real estate decision support
- End-to-end ML product development

## 4. Problem Statement
Predicting the price of a house accurately is important for real estate applications. This project solves that problem by using historical housing data and machine learning to estimate house value based on key characteristics.

## 5. Dataset Details
- Source: scikit-learn California Housing dataset
- Type: Tabular regression dataset
- Samples: 20,640 rows
- Features used:
  - MedInc: Median income of the neighborhood
  - HouseAge: Average house age
  - AveRooms: Average number of rooms
  - AveBedrms: Average number of bedrooms
  - Population: Total population of the area
  - AveOccup: Average household occupancy
  - Latitude: Geographic latitude
  - Longitude: Geographic longitude
- Target variable: Median house value, scaled in hundreds of thousands of dollars

## 6. Machine Learning Workflow
The project follows a standard ML pipeline:
1. Load the dataset
2. Explore the data
3. Split the data into training and test sets
4. Train a regression model
5. Evaluate model performance
6. Save the trained model and feature list
7. Deploy the model through FastAPI

## 7. Model Used
- Algorithm: RandomForestRegressor
- Reason for use: robust performance on structured/tabular data
- Training configuration:
  - n_estimators = 100
  - random_state = 42
- Output: predicted house price values

## 8. Model Evaluation
The model performance is evaluated using:
- Mean Absolute Error (MAE)
- R-squared Score (R2)

The API currently reports an estimated average error of about $39,000, which is used as a confidence range in the response.

## 9. Project Architecture
This project has two main parts:
- Machine Learning Part
  - Data loading
  - Training and evaluation
  - Model serialization using joblib
- API Part
  - FastAPI server
  - Input validation using Pydantic
  - Prediction endpoints for single and batch requests

## 10. File Structure
- main.py: FastAPI application and prediction endpoints
- train.py: model training and evaluation pipeline
- explore.py: data exploration and basic dataset inspection
- house_model.joblib: trained machine learning model
- house_features.joblib: list of input feature names
- readme.md: project documentation

## 11. API Endpoints
### Home endpoint
- GET /
- Returns a simple status message and endpoint guidance

### Health endpoint
- GET /health
- Returns service status and model information

### Single prediction endpoint
- POST /predict
- Accepts JSON input with house features
- Returns predicted house price in USD

### Batch prediction endpoint
- POST /predict-file
- Accepts a CSV file containing the required features
- Returns a CSV file with predicted prices

## 12. Input Schema
The API expects these fields:
- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

Each field is validated to ensure realistic and positive values.

## 13. Example Request
Example JSON body for the prediction endpoint:

```json
{
  "MedInc": 3.5,
  "HouseAge": 15,
  "AveRooms": 5.2,
  "AveBedrms": 1.0,
  "Population": 1200,
  "AveOccup": 3.0,
  "Latitude": 37.8,
  "Longitude": -122.2
}
```

## 14. How to Run the Project
### Install dependencies
Use a virtual environment and install the required packages:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic
```

### Train the model
```bash
python train.py
```

### Start the API server
```bash
uvicorn main:app --reload
```

Then open:
- http://127.0.0.1:8000/docs for the Swagger UI

## 15. Technologies Used
- Python
- FastAPI
- Uvicorn
- Pandas
- scikit-learn
- Pydantic
- Joblib
- JSON and CSV processing

## 16. Skills Demonstrated
This project highlights skills in:
- Machine learning model training
- Data preprocessing and analysis
- Regression modeling
- Model evaluation and metrics
- REST API development
- Input validation with Pydantic
- File upload and batch processing
- API documentation with Swagger
- Project documentation and portfolio presentation

## 17. Resume-Ready Summary
Built an end-to-end house price prediction project using Python, scikit-learn, and FastAPI. Trained a RandomForestRegressor model on the California Housing dataset, evaluated it with MAE and R2 metrics, and deployed it as a REST API with single and batch prediction endpoints. The project demonstrates practical experience in machine learning, model deployment, API design, and real-world problem solving.

## 18. Resume Bullet Points
- Developed a machine learning regression model to predict house prices from real-world housing data.
- Trained and evaluated a RandomForestRegressor using scikit-learn and validated performance with MAE and R2 metrics.
- Built a FastAPI-based web service for real-time house price predictions.
- Implemented both single-record and CSV-based batch prediction endpoints.
- Used Pydantic for request validation and joblib for model serialization.
- Created a professional project documentation structure suitable for GitHub and resume portfolio use.

## 19. Future Improvements
- Deploy the API to Render, Railway, or AWS
- Add authentication and user access control
- Dockerize the application
- Add CI/CD pipeline
- Improve the model with hyperparameter tuning
- Add frontend interface for easier usage

## 20. Revision Notes for Future Reference
When revisiting this project, focus on these key points:
- The project is an end-to-end ML deployment example
- It uses a tabular regression problem with real estate data
- The core model is RandomForestRegressor
- FastAPI is used to expose predictions as an API
- The project shows both ML and backend development skills
- It is suitable for demonstrating practical portfolio-ready work

---

## 21. Short Interview Answer
This project is an end-to-end machine learning and API deployment project. I used the California Housing dataset to train a regression model, evaluated it using MAE and R2, and deployed it with FastAPI so users can make predictions through a web API. It demonstrates my ability to move from data to deployment and present machine learning solutions as usable applications.



*Table of Content*
