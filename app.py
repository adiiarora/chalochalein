
import pickle
import joblib
import requests
from flask import Flask, request, render_template
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load models
hotel_model = joblib.load("hotel_model.pkl")
fare_model = joblib.load("fare_model.pkl")
cafe_model = joblib.load("cafe_model.pkl")  # using small model

# API key
WEATHER_API_KEY = "6df60cb2e9804fafe1af8942beee9159"

def get_temperature(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data['main']['temp']
    except:
        return 30

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        state = request.form['state']
        travel_date = request.form['travel_date']

        dt = datetime.strptime(travel_date, '%Y-%m-%d')
        month = dt.month
        day_of_week = dt.weekday()

        hotel_code = 111
        city_code = abs(hash(state)) % 100
        rating = 4.0
        reviews = 500
        star_rating = 4
        distance = 1000
        price_per_star = (rating * 1000) / star_rating

        hotel_input = pd.DataFrame([[hotel_code, rating, reviews, star_rating, distance, city_code, month, day_of_week, price_per_star]],
                                   columns=['Hotel Name', 'Rating', 'Reviews', 'Star Rating', 'Distance to Landmark', 'City', 'Month', 'DayOfWeek', 'PricePerStar'])
        hotel_price = hotel_model.predict(hotel_input)[0]

        temp = get_temperature(state)

        tourist_input = pd.DataFrame([{
            'interest': 'Historic',
            'latitude': 19.07,
            'longitude': 72.87,
            'google_rating': 4.2
        }])
        fare_class = int(fare_model.predict(tourist_input)[0])
        category_map = {0: 0, 1: 25, 2: 75, 3: 150}
        tourist_price = category_map[fare_class]

        # cafe model input must match training structure
        cafe_input = pd.DataFrame([[month, temp]], columns=["Month", "Temperature"])
        # add all cities used in one-hot encoding with 0s
        for col in cafe_model.feature_names_in_:
            if col.startswith("City_"):
                cafe_input[col] = 0
        # set selected state (converted to one-hot format)
        col_name = f"City_{state}"
        if col_name in cafe_input.columns:
            cafe_input[col_name] = 1

        cafe_price = cafe_model.predict(cafe_input)[0]

        return render_template('index.html', hotel=round(hotel_price), tourist=round(tourist_price), cafe=round(cafe_price), state=state.lower())

    except Exception as e:
        return f"Prediction error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
