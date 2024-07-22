import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn
import joblib
import logging
from streamlit_option_menu import option_menu


# Mappings for item types and status
item_type_mapping = {'W': 1, 'WI': 2, 'S': 3, 'Others': 4, 'PL': 5, 'IPL': 6, 'SLAWR': 7}
status_mapping = {
    'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4,
    'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8
}

status_options = ['Won', 'Draft', 'to be approved', 'lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product_options = ['611993', '611728', '628112', '628117', '628377', '640400', '640405', '640665', '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
        
# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models
#model_load_path1 = 'D:/Guvi_Data_Science/MDT33/Capstone_Project/Copper_Modelling/rfc_model2.joblib'
def predict_status(customer, country, item_type,
                   application, width, product_ref, thickness,  selling_price, quantity_tons):
    # Load the classification model
    with open("D:/Guvi_Data_Science/MDT33/Capstone_Project/Copper_Modelling/Class_model.pkl", "rb") as f1:
        Class_model = pickle.load(f1)
    # Ensure input data types are correct
    
    new_sample = np.array([[float(customer), float(country), float(item_type),
                                 float(application), float(width), float(product_ref),
                                 float(thickness), float(selling_price), float(quantity_tons)]])
        
    new_predict = Class_model.predict(new_sample)
        
    # Return the status based on prediction
    if new_predict[0]== 1:

        return "Won"
    else:
        return "Lose"
       
def predict_selling_price(quantity_tons, status, item_type,
                          application, thickness_log, width, customer, country, product_ref):
    
    # Load the regression model
    model_load_path1 = 'D:/Guvi_Data_Science/MDT33/Capstone_Project/Copper_Modelling/rfr_model3.joblib'
    RFR_model3 = joblib.load(model_load_path1)

    # Define the user data as a 2D numpy array
    y_pred = RFR_model3.predict(np.array([[np.log(quantity_tons), status, item_type, application,
                         np.log(thickness_log), width, country, customer, product_ref,0]]))

    # Convert the predicted log-transformed selling price back to the original scale
    selling_price = np.exp(y_pred[0])

    # Return the predicted selling price
    return selling_price


# Set page configuration
st.set_page_config(page_title="INDUSTRIAL COPPER MODELLING", page_icon="🏭", layout="wide")

# Create three columns for layout
col1, col2, col3 = st.columns([2, 4, 2])

# Add images to the left and right columns, and the title to the center column
with col1:
    st.image(r"D:\Guvi_Data_Science\MDT33\Capstone_Project\Copper_Modelling\Copper.jpg", width=350)  # Adjust the width as needed

with col2:
    st.markdown(
        """
        <h1 style='text-align: center; color: #00FF00;'>INDUSTRIAL 
        COPPER MODELLING 
        APPLICATION
        """,
        unsafe_allow_html=True
    )

with col3:
    st.image(r"D:\Guvi_Data_Science\MDT33\Capstone_Project\Copper_Modelling\Copper2.jpg", width=300)

# Function to set background gradient
def setting_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(to bottom right, #00008B, #ADD8E6); /* Replace colors with your preferred ones */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

setting_bg()

# Use tabs to separate the interface

options=option_menu("Predictions based On ML",["Predict Status", "Predict Selling Price"],
                        icons=['eye', 'calculator'],
                        menu_icon="cast",
                        default_index=0,
                        orientation="horizontal")

if options == "Predict Status":
    col1, col2, col3 = st.columns(3)
   
    with col1:
        citem_type = st.selectbox("Item Type", item_type_options, key=21)
        ccountry = st.selectbox('Country', sorted(country_options), key=31)
        capplication = st.selectbox('Application', sorted(application_options), key=41)
        cproduct_ref = st.selectbox("Product Reference", product_options, key=51)
        st.write("Minimum value=0.00001, Maximum value=6.807")
        cquantity = st.text_input("Enter the Quantity in Logs", value="0.00001")

    with col3:
        st.write("*Min. Value=5.983, Max. Value=7.391*")
        selling_price_log = st.text_input("**Enter the Selling Price**")
        st.write("Min. Value=1, Max. Value=2990")
        cwidth = st.text_input("Enter the Width", value="1")
        st.write("Minimum Value=12458, Maximum Value=30408185")
        ccustomer = st.text_input("Enter the Customer ID", value="12458")
        st.write("Minimum Value=0.16, Maximum Value=2.66")
        cthickness_log = st.text_input("Enter the Thickness", value="0.16")
        
    with col2:
        st.image(r"D:\Guvi_Data_Science\MDT33\Capstone_Project\Copper_Modelling\Won_lost.jpg", width=400)  # Fixed image file name
    
    with col2:
        button = st.button(":yellow[**PREDICT THE STATUS**]", use_container_width=True)

    if button:
        cquantity = float(cquantity)
        cthickness_log = float(cthickness_log)
        cwidth = float(cwidth)
        ccustomer = float(ccustomer)
        ccountry = float(ccountry)
        capplication = float(capplication)
        cproduct_ref = float(cproduct_ref)
        citem_type = item_type_mapping.get(citem_type)
        
    
        status = predict_status(ccustomer, ccountry, citem_type,
                            capplication, cwidth, cproduct_ref, cthickness_log, selling_price_log, cquantity)

        if status == "Won":

            st.markdown("<h3 style='text-align: center; color: Darkgreen; font-size: 24px; font-family: Arial;'>The Status is WON</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center; color: red; font-family: Arial;'>The Status is LOSE</h3>", unsafe_allow_html=True)
        
if options == "Predict Selling Price":

    col1, col2, col3 = st.columns(3)
    
    with col1:
        
        status = st.selectbox('Status', status_options, key=1)
        item_type = st.selectbox('Item Type', item_type_options, key=2)
        application = st.selectbox('Application', sorted(application_options), key=4)
        country = st.selectbox('Country', sorted(country_options), key=5)
        product_ref = st.selectbox("Product Reference", product_options, key=3)
    
    with col3:
        st.write("Minimum value=0.00001, Maximum value=6.807")
        quantity = st.text_input("Enter the Quantity in Logs")
        st.write("Minimum Value=0.16, Maximum Value=2.66")
        thickness_log = st.text_input("Enter the Thickness", value="0.16")
        st.write("Minimum Value=1, Maximum Value=2990")
        width = st.text_input("Enter the Width", value="1")
        st.write("Minimum Value=12458, Maximum Value=30408185")
        customer = st.text_input("Enter the Customer ID", value="12458")
        
    with col2:
        st.image(r"D:\Guvi_Data_Science\MDT33\Capstone_Project\Copper_Modelling\images.jpeg", width=400)

    with col2:
        
        button = st.button(":yellow[**PREDICT THE SELLING PRICE**]", use_container_width=True)

    if button:
        # Convert and preprocess inputs
        quantity = float(quantity)
        thickness_log = float(thickness_log)
        width = float(width)
        customer = float(customer)
        country = float(country)
        application = float(application)
        product_ref = float(product_ref)
        status = status_mapping.get(status)
        item_type = item_type_mapping.get(item_type)

           
        selling_price = predict_selling_price(quantity, status, item_type,
                                               application, thickness_log, width, customer, country, product_ref)
        
        st.markdown(f"<h3 style='text-align: center; color: green; font-family: Comic Sans MS;'>The Selling Price is: {selling_price:.2f}</h3>", unsafe_allow_html=True)    
                
