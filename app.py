import pandas as pd
from flask import Flask, render_template, request
import folium
import os
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
EXCEL_FILE = 'shopkeepers.xlsx'
df = pd.read_excel(EXCEL_FILE)
df['Achieved_Target'] = df['Achieved_Target'].fillna(0)
X = df[['Revenue', 'Target']]
y = df['Achieved_Target']
model = LogisticRegression()
model.fit(X, y)

df = df.sort_values(by='Revenue', ascending=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query'].strip().lower()
    results = df[(df['Area'].str.lower().str.contains(query)) | (df['Pincode'].astype(str).str.contains(query))]
    if not results.empty:
        results['Prediction'] = model.predict(results[['Revenue', 'Target']])
        results['Prediction_Label'] = results['Prediction'].apply(lambda x: '✅ Likely' if x == 1 else '❌ Unlikely')
    map_center = [results.iloc[0]['Latitude'], results.iloc[0]['Longitude']] if not results.empty else [26.9124, 75.7873]  # Jaipur coords default
    shop_map = folium.Map(location=map_center, zoom_start=13)

    for _, row in results.iterrows():
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=f"<b>{row['Shopkeeper_Name']}</b><br>Mobile: {row['Mobile_Number']}<br>Target: {row['Target']}<br>Revenue: ₹{row['Revenue']}<br>Prediction: {row['Prediction_Label']}"
        ).add_to(shop_map)

    map_html = shop_map._repr_html_()
    return render_template('results.html', results=results.to_dict(orient='records'), map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
