from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved models
rf_model = joblib.load("random_forest_model.pkl")
lin_model = joblib.load("linear_regression_model.pkl")

# Training columns (same as X_train)
feature_columns = ['age', 'bmi', 'children', 'smoker',
                   'sex_male',
                   'region_northwest', 'region_southeast', 'region_southwest']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Convert inputs into dataframe
        input_data = pd.DataFrame([[age, bmi, children, smoker, sex, region]],
                                  columns=['age', 'bmi', 'children', 'smoker', 'sex', 'region'])
        
        print(input_data)

        # Encode smoker and sex
        input_data['smoker'] = (input_data['smoker'] == 'yes')
        input_data['sex_male'] = (input_data['sex'] == 'male')
        input_data = input_data.drop(columns=['sex'])

        # One-hot encode region WITHOUT drop_first
        region_encoded = pd.get_dummies(input_data['region'], prefix='region')
        input_data = pd.concat([input_data.drop(columns=['region']), region_encoded], axis=1)

        # Reindex to ensure all training columns exist
        input_encoded = input_data.reindex(columns=feature_columns, fill_value=0)

        print(input_encoded)

        # Predict with both models
        pred_rf = rf_model.predict(input_encoded)[0]
        pred_lin = lin_model.predict(input_encoded)[0]

        print(input_encoded)
        print(pred_rf, pred_lin)

        return render_template('index.html',
                               prediction_rf=round(pred_rf, 2),
                               prediction_lin=round(pred_lin, 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
