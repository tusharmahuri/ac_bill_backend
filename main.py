from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the saved model and encoder
with open('ac_calculator_model_updated_17_12-24.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('ac_calculator_encoder_updated_17_12_24.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Electricity rates by state
ele_rate = {
  "Andaman & Nicobar Island": 752,
  "Andhra Pradesh": 860,
  "Arunachal Pradesh": 400,
  "Assam": 810,
  "Bihar (Urban Areas)": 825,
  "Bihar (Rural Areas)": 705,
  "Chandigarh": 441,
  "Chhattisgarh": 670,
  "Dadra & Nagar Haveli": 303,
  "Daman & Diu": 290,
  "Delhi-(BYPL/BRPL/TPDDL)": 638,
  "Delhi-(NDMC)": 638,
  "Goa": 400,
  "Gujarat (Urban Areas)": 558,
  "Gujarat-(Torrent Power Ltd., Ahmedabad)": 549,
  "Gujarat-(Torrent Power Ltd., Surat)": 549,
  "Haryana": 560,
  "Himachal Pradesh": 555,
  "Jammu & Kashmir": 436,
  "Jharkhand (Urban Areas)": 672,
  "Jharkhand (Rural Areas)": 612,
  "Karnataka": 943,
  "Kerala": 1160,
  "Ladakh": 385,
  "Lakshadweep": 588,
  "Madhya Pradesh (Urban Areas)": 910,
  "Madhya Pradesh (Rural Areas)": 960,
  "Maharashtra": 1250,
  "Maharashtra-Mumbai (B.E.S.T)": 957,
  "Maharashtra-Mumbai-Adani Electricity": 840,
  "Maharashtra-Mumbai 'TATA's": 901,
  "Manipur": 708,
  "Meghalaya": 815,
  "Mizoram": 630,
  "Nagaland": 668,
  "Odisha": 606,
  "Puducherry": 556,
  "Punjab": 886,
  "Rajasthan": 843,
  "Sikkim": 344,
  "Tamil Nadu": 822,
  "Telangana": 945,
  "Tripura": 753,
  "Uttarakhand": 656,
  "Uttar Pradesh (Urban)": 768,
  "Uttar Pradesh (Rural)": 613,
  "West Bengal (Urban)": 951,
  "West Bengal (Rural)": 606,
  "D.V.C. (Jharkhand Area)": 460
}

import pymongo
# MongoDB connection string
connection_string = "mongodb+srv://flash:ykF2wCPfdiGH0agr@cluster0.fkkr3.mongodb.net/acbill?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = pymongo.MongoClient(connection_string)

# Specify the database and collection
database_name = "acbill"  # Replace with your database name
collection_name = "products"  # Replace with your collection name
db = client[database_name]
collection = db[collection_name]

# Fetch data from the collection
cursor = collection.find()  # Get all documents
data = list(cursor)  # Convert cursor to a list of documents

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Optionally, drop the MongoDB-specific '_id' field if not needed
if '_id' in df.columns:
    df.drop(columns=['_id'], inplace=True)
df['Annual Consumption']= 900 #this should be removed
df['type'] = df['type'].replace('Split', 'Split AC')
df = df[['brand', 'capacity', 'type', 'starRating', 'price', 'tag', 'image', 'productLink', 'productSource', 'Annual Consumption']]
column_mapping = {
    'brand': 'Brand',
    'capacity': 'Capacity',
    'type': 'AC Type',
    'starRating': 'Star Rating'
}
df.rename(columns=column_mapping, inplace=True)
# FastAPI app
app = FastAPI()

# Request body schema
class ACInput(BaseModel):
    brand: str
    capacity: float
    starRating: int
    type: str
    temperature: str
    totalUsagesHour: int
    state: str

def calculate_range(value):
    lower_bound = round(value * 0.95)
    upper_bound = round(value * 1.05)
    return f"{lower_bound}-{upper_bound}"

def recommend_ac(user_input, df, top_n=10):
    # Extract user input
    brand = user_input.get('Brand', '').strip().lower()
    capacity = user_input.get('Capacity', None)
    ac_type = user_input.get('AC Type', '').strip().lower()
    star_rating = user_input.get('Star Rating', None)

    # Normalize dataframe columns for case-insensitive matching
    df['Brand_lower'] = df['Brand'].str.lower()
    df['AC Type_lower'] = df['AC Type'].str.lower()

    # Filter the dataframe based on user preferences for capacity, AC type, and star rating
    filtered_df = df[
        (df['Capacity'] == capacity) &
        (df['AC Type_lower'] == ac_type) &
        (df['Star Rating'] == star_rating)
        ]

    # Sort the recommendations by Annual Consumption (lower to higher)
    filtered_df = filtered_df.sort_values(by='Annual Consumption', ascending=True)

    # Prioritize the input brand by splitting the dataframe
    brand_df = filtered_df[filtered_df['Brand_lower'] == brand]
    other_brands_df = filtered_df[filtered_df['Brand_lower'] != brand]

    # Concatenate the input brand's details at the top, followed by other brands
    recommended_df = pd.concat([brand_df, other_brands_df])

    # Drop the helper columns and return the top N recommendations
    recommended_df = recommended_df.drop(columns=['Brand_lower', 'AC Type_lower'])
    recommended_df = recommended_df.head(top_n)
    return recommended_df

@app.post("/predict-and-recommend")
async def predict_and_recommend(data: ACInput):
    # Extract inputs
    brand = data.brand
    capacity = data.capacity
    star_rating = data.starRating
    ac_type = data.type
    temp_pref = data.temperature
    hours_usage = data.totalUsagesHour
    state = data.state

    # Encode features
    input_features = pd.DataFrame([[brand, capacity, star_rating, ac_type]],
                                   columns=['Brand', 'Capacity', 'Star Ratting', 'AC Type'])
    try:
        encoded_features = encoder.transform(input_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Encoding failed: {e}")

    # Predict annual consumption
    annual_consumption = model.predict(encoded_features)[0]
    annual_consumption_with_error = annual_consumption * 1.2  # Add 20% error

    # Adjust for temperature preference
    if temp_pref == '16-20':
        annual_consumption_actual = annual_consumption_with_error * 1.35
    elif temp_pref == '20-22':
        annual_consumption_actual = annual_consumption_with_error * 1.20
    elif temp_pref == '24':
        annual_consumption_actual = annual_consumption_with_error
    else:  # >24
        annual_consumption_actual = annual_consumption_with_error * 0.90

    # Calculate hourly, daily, monthly, and yearly consumption
    hourly_consumption = annual_consumption_actual / 1920  # 8 hours/day for 8 months
    daily_consumption = hourly_consumption * hours_usage
    monthly_consumption = daily_consumption * 30
    yearly_consumption = monthly_consumption * 8

    # Calculate costs
    unit_price = ele_rate.get(state, 0) / 100  # Convert to price from paisa
    if unit_price == 0:
        raise HTTPException(status_code=404, detail="State not found in electricity rates.")
    daily_cost = daily_consumption * unit_price
    monthly_cost = monthly_consumption * unit_price
    yearly_cost = yearly_consumption * unit_price

    # Get Recommendations
    user_input = {
        "Brand": brand,
        "Capacity": capacity,
        "Star Rating": star_rating,
        "AC Type": ac_type
    }
    recommended_acs = recommend_ac(user_input, df, top_n=5)
    recommendations = []
    for _, row in recommended_acs.iterrows():
        rec_annual_cost = (row["Annual Consumption"] / 1920 * hours_usage * 30 * 8) * unit_price
        recommendations.append({
            "Image": row['image'],
            "starRating": row["Star Rating"],
            "tagline": row["tag"],
            "price": row["price"],  # Mock data; adjust with pricing if available
            "brand": row["Brand"],
            "capacity": row["Capacity"],
            "type": row["AC Type"],
            "estimatedMonthlyCost": calculate_range(round(rec_annual_cost / 8)),
            "buttonText": row["productSource"],
            "ProductLink": row["productLink"]
        })

    # Response
    response = {
        "predictions": {
            "brand": brand,
            "capacity": capacity,
            "type": ac_type,
            "starRating": star_rating,
            "temperature": temp_pref,
            "totalUsagesHour": hours_usage,
            "monthlyUnitConsumption": calculate_range(round(monthly_consumption)),
            "monthlyCost": calculate_range(round(monthly_cost)),
            "unitPrice": unit_price,
            "dailyCost": calculate_range(round(daily_cost)),
            "yearlyCost": calculate_range(round(yearly_cost))
        },
        "chatData": {
            "userHourlyUsage": hours_usage,  # user's input for total usage hours
            "averageUserHourlyUsage": 8  # Fixed value as per your request
        },
        "recommendations": recommendations
    }

    return response
