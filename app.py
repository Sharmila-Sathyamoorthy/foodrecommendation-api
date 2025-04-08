from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from fuzzywuzzy import fuzz, process

app = Flask(__name__)
CORS(app)

# Load the dataset
food_dataset = pd.read_excel("final food db.xlsx")

# Define features and target variables
features = ['Food_items', 'Breakfast', 'Lunch', 'Dinner', 'VegNovVeg']
target = ['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Carbohydrates', 'Fibre', 'VitaminD']
target_recommend = 'Food_Style_Comment_and_Recommendation'

# Preprocessing
X = food_dataset[features].copy()
y = food_dataset[target]
y_recommend = food_dataset[target_recommend]

label_encoder_food = LabelEncoder()
X.loc[:, 'Food_items'] = label_encoder_food.fit_transform(X['Food_items'])

label_encoder_veg = LabelEncoder()
all_veg_values = pd.concat([food_dataset['VegNovVeg'].astype(str), pd.Series(['Veg', 'veg', 'Non-veg', 'non-veg'])], ignore_index=True)
label_encoder_veg.fit(all_veg_values)
X.loc[:, 'VegNovVeg'] = label_encoder_veg.transform(X['VegNovVeg'].astype(str))

sc = StandardScaler()
x = sc.fit_transform(X[['Food_items', 'VegNovVeg']])

X_train, X_test, y_train, y_test, y_train_recommend, y_test_recommend = train_test_split(
    x, y, y_recommend, test_size=0.2, random_state=2
)

model = RandomForestRegressor(random_state=2)
model.fit(X_train, y_train)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(y_recommend)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    food_items_with_quantities = data["food_items"]
    gender = data["gender"]

    total_nutrient_predictions = np.zeros(len(target))
    food_style_comments = []

    for food_item_with_quantity in food_items_with_quantities:
        match = re.match(r"([a-zA-Z\s]+)(\d+)?", food_item_with_quantity)
        if match:
            food_item = match.group(1).strip()
            quantity = int(match.group(2)) if match.group(2) else 1
        else:
            continue

        closest_match = process.extractOne(food_item, food_dataset['Food_items'], scorer=fuzz.token_sort_ratio)

        if closest_match and closest_match[1] >= 50:
            food_item = closest_match[0]
        else:
            continue

        input_data = {
            'Food_items': food_item,
            'VegNovVeg': 'Veg' if food_item in food_dataset[food_dataset['VegNovVeg'].isin(['Veg', 'veg'])]['Food_items'].values else 'Non-veg'
        }

        input_features = pd.DataFrame([input_data], columns=['Food_items', 'VegNovVeg'])

        try:
            input_features.loc[:, 'Food_items'] = label_encoder_food.transform(input_features['Food_items'])
        except ValueError:
            continue

        input_features.loc[:, 'VegNovVeg'] = label_encoder_veg.transform(input_features['VegNovVeg'].astype(str))
        input_features_scaled = sc.transform(input_features[['Food_items', 'VegNovVeg']])
        nutrient_predictions = model.predict(input_features_scaled)[0]
        nutrient_predictions = nutrient_predictions * quantity
        total_nutrient_predictions += nutrient_predictions

        if food_item:
            food_index = label_encoder_food.transform([food_item])[0]
            food_style_comment = food_dataset.loc[food_index, 'Food_Style_Comment_and_Recommendation']
            food_style_comments.extend([food_style_comment] * quantity)

    predicted_nutrients = dict(zip(target, total_nutrient_predictions))
    review = ""
    if gender == "M" and predicted_nutrients['Calories'] > 3000:
        review = "You crossed your calorie limit."
    elif gender == "F" and predicted_nutrients['Calories'] > 2400:
        review = "You crossed your calorie limit."

    recommendation = Counter(food_style_comments).most_common(1)[0][0] if food_style_comments else "No recommendations found."

    return jsonify({
        "nutrient_summary": str(predicted_nutrients),
        "review": review,
        "recommendation": recommendation
    })
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
