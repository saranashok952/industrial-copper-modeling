import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu
import joblib

# Functions
def predict_status(country_data,item_type_data,application_data,width_data,product_ref_data,quality_tons_log_data,customer_log_data,thickness_log_data,selling_price_log_data,item_date_day_data,item_date_month_data,item_date_year_data,delivery_date_day_data,delivery_date_month_data,delivery_date_year_data):

    #change the datatypes "string" to "int"
    item_date_day_data= int(item_date_day_data)
    item_date_month_data= int(item_date_month_data)
    item_date_year_data= int(item_date_year_data)

    delivery_date_day_data= int(delivery_date_day_data)
    delivery_date_month_data= int(delivery_date_month_data)
    delivery_date_year_data= int(delivery_date_year_data)
    #modelfile of the classification
    with open("c:/Users/dell/Downloads/model_save_classification","rb") as f:
        model_class=joblib.load(f)

    user_data= np.array([[country_data,item_type_data,application_data,width_data,product_ref_data,quality_tons_log_data,customer_log_data,thickness_log_data,
                       selling_price_log_data,item_date_day_data,item_date_month_data,item_date_year_data,delivery_date_day_data,delivery_date_month_data,delivery_date_year_data]])
    
    y_pred= model_class.predict(user_data)

    if y_pred == 1:
        return 1
    else:
        return 0

def predict_selling_price(country_data,status_data,item_type_data,application_data,width_data,product_ref_data,quality_tons_log_data,customer_log_data,
                   thickness_log_data,item_date_day_data,item_date_month_data,item_date_year_data,delivery_date_day_data,delivery_date_month_data,delivery_date_year_data):

    #change the datatypes "string" to "int"
    item_date_day_data= int(item_date_day_data)
    item_date_month_data= int(item_date_month_data)
    item_date_year_data= int(item_date_year_data)

    delivery_date_day_data= int(delivery_date_day_data)
    delivery_date_month_data= int(delivery_date_month_data)
    delivery_date_year_data= int(delivery_date_year_data)
    #modelfile of the classification
    with open("C:/Users/dell/Downloads/model_save_regression","rb") as f:
        model_regg=joblib.load(f)

    user_data= np.array([[country_data,status_data,item_type_data,application_data,width_data,product_ref_data,quality_tons_log_data,customer_log_data,thickness_log_data,
                       item_date_day_data,item_date_month_data,item_date_year_data,delivery_date_day_data,delivery_date_month_data,delivery_date_year_data]])
    
    y_pred= model_regg.predict(user_data)

    ac_y_pred= np.exp(y_pred[0])

    return ac_y_pred


st.set_page_config(layout= "wide")

st.title(":blue[**INDUSTRIAL COPPER MODELING**]")

option = option_menu(
    menu_title = None,
    options = ["HOME", "PREDICT STATUS", "PREDICT SELLING PRICE"],
    icons =["house","graph-up-arrow","currency-exchange"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": " #7FD8BE","size":"cover", "width": "200"},
        "icon": {"color": "black", "font-size": "25px"},

        "nav-link": {"font-size": "25px", "text-align": "center", "margin": "-2px", "--hover-color": " #FCEFEF"},
        "nav-link-selected": {"background-color": "#FF7F50",  "font-family": "YourFontFamily"}})
  
if option == "HOME":

    st.markdown(
    "<h1 style='text-align: center;'></h1>",
    unsafe_allow_html=True,
)

# Then display your image
    st.header("Overview:")

    st.write('''The project aims to address challenges faced in the copper industry, including dealing with skewed and noisy data, optimizing pricing decisions, and developing a lead classification model to evaluate potential customers. The project utilizes Python programming language and various libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit.''')
    
    st.header("Key Features:")

    st.write('''Data understanding and preprocessing techniques''')
    st.write('''Exploratory Data Analysis (EDA) with visualization''')
    st.write('''Machine learning modeling for regression and classification tasks''')
    st.write('''Development of an interactive web application using Streamlit''')
    st.write('''Modular code development for maintainability and portability''')
    

if option == "PREDICT STATUS":

    st.header("STATUS (Won / Lose)")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:
        country= st.number_input(label="**COUNTRY** (25.0-113.0)")
        item_type= st.number_input(label="**ITEM TYPE** (0.0-6.0)")
        application= st.number_input(label="**APPLICATION** (2.0-87.5)")
        width= st.number_input(label="**WIDTH** (700.0-1980.0)")
        product_ref= st.number_input(label="**PRODUCT REF** (611728-1722207579)")
        quantity_tons_log= st.number_input(label="**QUANTITY TONS LOG** (-0.322-6.924)",format="%0.15f")
        customer_log= st.number_input(label="**CUSTOMER LOG** (17.21910-:17.23015)",format="%0.15f")
        thickness_log= st.number_input(label="**THICKNESS LOG** (-1.71479-3.28154)",format="%0.15f")
    
    with col2:
        selling_price_log= st.number_input(label="**SELLING PRICE LOG** (5.97503-7.39036)",format="%0.15f")
        item_date_day= st.selectbox("**ITEM DATE DAY**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month= st.selectbox("**ITEM DATE MONTH**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year= st.selectbox("**ITEM DATE YEAR**",("2020","2021"))
        delivery_date_day= st.selectbox("**DELIVERY DATE DAY**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month= st.selectbox("**DELIVERY DATE MONTH**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year= st.selectbox("**DELIVERY DATE YEAR**",("2020","2021","2022"))
        

    button= st.button(":violet[***PREDICT THE STATUS***]",use_container_width=True)

    if button:
        status= predict_status(country,item_type,application,width,product_ref,quantity_tons_log,
                               customer_log,thickness_log,selling_price_log,item_date_day,
                               item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                               delivery_date_year)
        
        if status == 1:
            st.write("## :green[**WON**]")
        else:
            st.write("## :red[**LOSE**]")

if option == "PREDICT SELLING PRICE":

    st.header("**SELLING PRICE**")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:
        country= st.number_input(label="**COUNTRY** (25.0-113.0)")
        status= st.number_input(label="**STATUS** (0.0-8.0)")
        item_type= st.number_input(label="**ITEM TYPE** (0.0-6.0)")
        application= st.number_input(label="**APPLICATION** (2.0-87.5)")
        width= st.number_input(label="**WIDTH** (700.0-1980.0)")
        product_ref= st.number_input(label="**PRODUCT REF** (611728-1722207579)")
        quantity_tons_log= st.number_input(label="**QUANTITY TONS LOG** (-0.322-6.924)",format="%0.15f")
        customer_log= st.number_input(label="**CUSTOMER LOG** (17.21910-:17.23015)",format="%0.15f")
        
    
    with col2:
        thickness_log= st.number_input(label="**THICKNESS LOG** (-1.71479-3.28154)",format="%0.15f")
        item_date_day= st.selectbox("**ITEM DATE DAY**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month= st.selectbox("**ITEM DATE MONTH**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year= st.selectbox("**ITEM DATE YEAR**",("2020","2021"))
        delivery_date_day= st.selectbox("**DELIVERY DATE DAY**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month= st.selectbox("**DELIVERY DATE MONTH**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year= st.selectbox("**DELIVERY DATE YEAR**",("2020","2021","2022"))
        

    button= st.button(":violet[***PREDICT THE SELLING PRICE***]",use_container_width=True)

    if button:
        price= predict_selling_price(country,status,item_type,application,width,product_ref,quantity_tons_log,
                               customer_log,thickness_log,item_date_day,
                               item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                               delivery_date_year)
        
        
        st.write("## :green[**Selling Price:**]",price)