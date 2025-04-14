from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the pickled model
model = joblib.load("model/model.pkl")
data = pd.read_csv("data/user_rating_detail.csv")
all_places = data[
    ["Place_Id", "Place_Name", "Category", "Description", "Lat", "Long"]
].drop_duplicates("Place_Id")
all_place_ids = all_places["Place_Id"].unique()


@app.get("/predict/{place_id}")
def predict(place_id: int):
    predictions = []
    try:

        matching_users = data[
            (data["Place_Id"] == place_id) & (data["Place_Ratings"] == 5)
        ]

        if not matching_users.empty:
            user_id = matching_users.sample(1)["User_Id"].values[0]
        else:
            user_id = 1

        places_to_predict = [
            p for p in all_place_ids if p not in matching_users["Place_Id"].values
        ]

        user_input = np.array([user_id] * len(places_to_predict))
        place_input = np.array(places_to_predict)

        batch_predictions = model.predict([user_input, place_input]).flatten()

        for i, pid in enumerate(places_to_predict):
            predictions.append(
                {
                    "user_id": user_id,
                    "place_id": pid,
                    "predicted_rating": batch_predictions[i],
                }
            )

        predictions_df = pd.DataFrame(predictions)

        top_places = predictions_df.sort_values(
            "predicted_rating", ascending=False
        ).head(20)
        recommendations = pd.merge(
            top_places,
            all_places[
                ["Place_Id", "Place_Name", "Category", "Description", "Lat", "Long"]
            ],
            left_on="place_id",
            right_on="Place_Id",
        ).drop("Place_Id", axis=1)

        return {
            "predictions": recommendations.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
