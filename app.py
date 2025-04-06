from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz, process
from collections import Counter
import joblib

app = Flask(__name__)

# Load dataset
food_dataset = pd.read_excel("final food db.xlsx")

features = ['Food_items', 'Breakfast', 'Lunch', 'Dinner', 'VegNovVeg']
target = ['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Carbohydrates', 'Fibre', 'VitaminD']
target_recommend = 'Food_Style_Comment_and_Recommendation'

X = food_dataset[features].copy()
y = food_dataset[target]
y_recommend = food_dataset[target_recommend]

label_encoder_food = LabelEncoder()
X['Food_items'] = label_encoder_food.fit_transform(X['Food_items'])

label_encoder_veg = LabelEncoder()
all_veg_values = pd.concat([food_dataset['VegNovVeg'].astype(str), pd.Series(['Veg', 'veg', 'Non-veg', 'non-veg'])])
label_encoder_veg.fit(all_veg_values)
X['VegNovVeg'] = label_encoder_veg.transform(X['VegNovVeg'].astype(str))

sc = StandardScaler()
x = sc.fit_transform(X[['Food_items', 'VegNovVeg']])

model = RandomForestRegressor(random_state=2)
model.fit(x, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    food_items_with_quantities = data['food_items']
    gender = data.get('gender', 'M')
    total_nutrient_predictions = np.zeros(len(target))
    food_style_comments = []

    for item in food_items_with_quantities:
        match = re.match(r"([a-zA-Z\s]+)(\d+)?", item)
        if match:
            food_item = match.group(1).strip()
            quantity = int(match.group(2)) if match.group(2) else 1
        else:
            food_item = item
            quantity = 1

        closest_match = process.extractOne(food_item, food_dataset['Food_items'], scorer=fuzz.token_sort_ratio)
        if not closest_match or closest_match[1] < 50:
            continue
        food_item = closest_match[0]

        input_data = {
            'Food_items': food_item,
            'VegNovVeg': 'Veg' if food_item in food_dataset[food_dataset['VegNovVeg'].isin(['Veg', 'veg'])]['Food_items'].values else 'Non-veg'
        }

        try:
            input_df = pd.DataFrame([input_data])
            input_df['Food_items'] = label_encoder_food.transform(input_df['Food_items'])
            input_df['VegNovVeg'] = label_encoder_veg.transform(input_df['VegNovVeg'].astype(str))
        except Exception as e:
            continue

        scaled_input = sc.transform(input_df[['Food_items', 'VegNovVeg']])
        prediction = model.predict(scaled_input)[0] * quantity
        total_nutrient_predictions += prediction

        index = food_dataset[food_dataset['Food_items'] == food_item].index[0]
        food_style_comments.append(food_dataset.loc[index, 'Food_Style_Comment_and_Recommendation'])

    nutrient_dict = dict(zip(target, total_nutrient_predictions))
    exceeded = (gender == "M" and nutrient_dict["Calories"] > 3000) or (gender == "F" and nutrient_dict["Calories"] > 2400)

    return jsonify({
        "nutrients": nutrient_dict,
        "exceeded_limit": exceeded,
        "recommendation": Counter(food_style_comments).most_common(1)[0][0] if food_style_comments else "No recommendation found."
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

