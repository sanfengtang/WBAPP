import gradio as gr
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# load the median values of the features
medians = pd.read_csv('C:\\medians.csv')

# original df column names
df = pd.read_csv('C:\\web_data.csv', encoding='gbk')

# selected features
features = ['Manufacturer', 'Model', 'Mileage', 'New car price', 'Number of seats', 'Max. horsepower(Ps)', 'Displacement', 'Integrated fuel consumption(L/100km)']

# Function to model data to fit the model
def transform(data):
    sc = StandardScaler()
    sr_reg = pickle.load(open('C:\\model_sr.pkl','rb'))

    # create a new DataFrame with the medians
    new_df = pd.DataFrame(medians)

    # replace the values of the selected features with the user input
    for i, feature in enumerate(features):
        new_df[feature] = data[i]

    X_new = sc.fit_transform(new_df)
    output = sr_reg.predict(X_new)
    return "The price of the car is " + str(round(np.exp(output)[0],2)) + "$"

# main function to predict the price
def predict_price(manufacturer, model, mileage, new_car_price, num_of_seats, max_horsepower, displacement, fuel_consumption): 
    new_data = [manufacturer, model, mileage, new_car_price, num_of_seats, max_horsepower, displacement, fuel_consumption]
    return transform(new_data) 

# create the inputs for the gradio interface
inputs = []
inputs.append(gr.inputs.Dropdown(choices=df['Manufacturer'].unique().tolist(), label='Manufacturer'))
inputs.append(gr.inputs.Dropdown(choices=df['Model'].unique().tolist(), label='Model'))
inputs.append(gr.inputs.Textbox(label='Mileage'))
inputs.append(gr.inputs.Textbox(label='New car price'))
inputs.append(gr.inputs.Dropdown(choices=[2, 3, 4, 5, 6, 7, 8], label='Number of seats'))
inputs.append(gr.inputs.Slider(minimum=0, maximum=1000, label='Max. horsepower(Ps)'))
inputs.append(gr.inputs.Slider(minimum=0, maximum=10, label='Displacement(L)'))
inputs.append(gr.inputs.Slider(minimum=0, maximum=20, label='Integrated fuel consumption(L/100km)'))

app = gr.Interface(title="Predict the price of a used car based on its specs", 
                    fn=predict_price,
                    inputs=inputs,
                    outputs="text")

app.launch()
