import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import os
import base64


#load models
#1. FOR BANK DEPOSIT PREDICTION
#random forest classifier
RF_pickle_in = open("FD-RandomForestclassifier.pkl","rb")
RF_classifier =pickle.load(RF_pickle_in)

#decision tree model

# DT_pickle_in = open("FD-DecisionTreeClassifier.pkl","rb")
# DT_classifier=pickle.load(DT_pickle_in)

#KNN model

# KNN_pickle_in = open("FD-KNNClassifier.pkl","rb")
# KNN_classifier=pickle.load(KNN_pickle_in)

#2. TO CHECK IF CUSTOMER HAS A CREDIT BALANCE OR NOT
C_RF_pickle_in = open("Cred-RFclassifier.pkl","rb")
C_RF_classifier =pickle.load(C_RF_pickle_in)

#decision tree model

# C_DT_pickle_in = open("Cred-DTClassifier.pkl","rb")
# C_DT_classifier=pickle.load(C_DT_pickle_in)

#KNN model

# C_KNN_pickle_in = open("Cred-KNNClassifier.pkl","rb")
# C_KNN_classifier=pickle.load(C_KNN_pickle_in)

# #3. TO CHECK IF CUSTOMER WILL TAKE A HOUSING LOAN OR NOT
 H_RF_pickle_in = open("H-RandomForestC.pkl","rb")
 H_RF_classifier =pickle.load(H_RF_pickle_in)

#decision tree model

# H_DT_pickle_in = open("H-DTClassifier.pkl","rb")
# H_DT_classifier=pickle.load(H_DT_pickle_in)

# #KNN model

# H_KNN_pickle_in = open("H-KNNClassifier.pkl","rb")
# H_KNN_classifier=pickle.load(H_KNN_pickle_in)

#4. TO CHECK IF CUSTOMER WILL TAKE A PERSONAL LOAN OR NOT

P_RF_pickle_in = open("PL_RandomForestclassifier.pkl","rb")
P_RF_classifier =pickle.load(P_RF_pickle_in)

#decision tree model

# P_DT_pickle_in = open("PL_DecisionTreeClassifier.pkl","rb")
# P_DT_classifier=pickle.load(P_DT_pickle_in)

# #KNN model

# P_KNN_pickle_in = open("PL_KNNClassifier.pkl","rb")
# P_KNN_classifier=pickle.load(P_KNN_pickle_in)

#allocate number for select options
month_dict = {'January': 1, 'February': 2,'March':3,'April': 4,'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9,'October':10,'November': 11,'December': 12}
feature_dict = {"No":0,"Yes":1}
job_dict = {'blue-collar': 1, 'management': 2,'technician':3,'admin.': 4,'services': 5, 'retired': 6, 'self-employed': 7, 'entrepreneur': 8, 'unemployed': 9,'housemaid':10,'student': 11,'unknown': 12}
edu_dict = {'primary': 1, 'secondary': 2,'tertiary':3,'unknown': 0} 

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 





def main():
    st.title("Bank Data Analysis 🏦")
    st.sidebar.title("Bank Data Analysis 🏦")
    st.markdown("This Application is for Bank Data Analysis")
    st.sidebar.markdown("The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls.")
    activities = ["Home","Visualize Data", "Predict Fixed Deposit Creation", "Loan Granting", "Predict Housing Loan Necessity", "Predict Personal Loan Necessity"]
    choice = st.sidebar.selectbox("Choose activity", activities)

    if choice == 'Predict Fixed Deposit Creation':
        st.info("Prediction if Customer creates fixed deposit or not")
        age = st.number_input("Age",18,100)
        job   = st.selectbox("Occupation", tuple(job_dict.keys()) )
        housing	 = st.radio("Has housing loan?", tuple(feature_dict.keys()) )
        month	 = st.selectbox("Whiich month was the client last contacted in? ",tuple(month_dict.keys()) )
        duration  = st.number_input("Duration of last call in seconds", 1, 5000)
        pdays = st.text_input("Number of days that passed by after the client was last contacted(-1 if not contacted) " )
        previous = st.text_input("Number of contacts performed before this campaign and for this client " )

        feature_list = [age,get_value(job,job_dict),get_fvalue(housing),get_value(month, month_dict),duration,pdays, previous]
        single_sample = np.array(feature_list).reshape(1,-1)


        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        st.text("")
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
                prediction = RF_classifier.predict(single_sample)
                pred_prob = RF_classifier.predict_proba(single_sample)
            # elif model_choice == "Decision Tree Classifier":
            #     prediction = DT_classifier.predict(single_sample)
            #     pred_prob = DT_classifier.predict_proba(single_sample)
            # else:
            #     prediction = KNN_classifier.predict(single_sample)
            #     pred_prob = KNN_classifier.predict_proba(single_sample)

            if prediction == 0:
                st.text("")
                st.warning("Customer doesn't create Bank Term Deposit")
                pred_probability_score = {"Not creating account":pred_prob[0][0]*100,"Creating Account":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.info(pred_probability_score)
                	
							
            else:
                st.text("")
                st.success("Customer creates Bank Term Deposit")
                pred_probability_scoreY = {"Not creating account":pred_prob[0][0]*100,"Creating Account":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.json(pred_probability_scoreY)    
        
    


    elif choice == 'Loan Granting':
        st.info("Prediction if Customer is suitable to grant bank loan")
        age = st.number_input("Age",18,100)
        job   = st.selectbox("Occupation", tuple(job_dict.keys()) )
        balance = st.text_input("Savings account balance:")
        loan	 = st.radio("Has taken personal loan?", tuple(feature_dict.keys()) )
        day = st.number_input("Which day of the month were they contacted?", 1, 31)
        month	 = st.selectbox("Whiich month was the client last contacted in? ",tuple(month_dict.keys()) )
        pdays = st.text_input("Number of days that passed by after the client was last contacted(-1 if not contacted) " )
        previous = st.text_input("Number of contacts performed before this campaign and for this client " )
        ifCreatedAccount	 = st.radio("Has a Fixed Deposit Account?", tuple(feature_dict.keys()) )

        feature_list = [age,get_value(job,job_dict),balance,get_fvalue(loan), day, get_value(month, month_dict),pdays, previous, get_fvalue(ifCreatedAccount)]
        single_sample = np.array(feature_list).reshape(1,-1)


        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        st.text("")
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
                prediction = C_RF_classifier.predict(single_sample)
                pred_prob = C_RF_classifier.predict_proba(single_sample)
            # elif model_choice == "Decision Tree Classifier":
            #     prediction = C_DT_classifier.predict(single_sample)
            #     pred_prob = C_DT_classifier.predict_proba(single_sample)
            # else:
            #     prediction = C_KNN_classifier.predict(single_sample)
            #     pred_prob = C_KNN_classifier.predict_proba(single_sample)

            if prediction == 1:
                st.text("")
                st.warning("Customer isn't suitable to give a loan")
                C_pred_probability_score = {"Won't have a credit balance":pred_prob[0][0]*100,"Will have a credit balance":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.info(C_pred_probability_score)
                	
							
            else:
                st.text("")
                st.success("Customer is suitable to give a loan")
                C_pred_probability_scoreY = {"Won't have a credit balance":pred_prob[0][0]*100,"Will have a credit balance":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.json(C_pred_probability_scoreY)    

    elif choice == 'Predict Housing Loan Necessity':
        st.info("Prediction if Customer will need a Housing Loan")
        age = st.number_input("Age",18,100)
        job   = st.selectbox("Occupation", tuple(job_dict.keys()) )
        education = st.selectbox("Education", tuple(edu_dict.keys()))
        balance = st.text_input("Savings account balance:")
        month	 = st.selectbox("Whiich month was the client last contacted in? ",tuple(month_dict.keys()) )
        pdays = st.text_input("Number of days that passed by after the client was last contacted(-1 if not contacted) " )
        ifCreatedAccount	 = st.radio("Has a Fixed Deposit Account?", tuple(feature_dict.keys()) )

        feature_list = [age,get_value(job,job_dict),get_value(education, edu_dict), balance, get_value(month, month_dict),pdays,get_fvalue(ifCreatedAccount)]
        single_sample = np.array(feature_list).reshape(1,-1)


        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        st.text("")
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
                prediction = H_RF_classifier.predict(single_sample)
                pred_prob = H_RF_classifier.predict_proba(single_sample)
            # elif model_choice == "Decision Tree Classifier":
            #     prediction = H_DT_classifier.predict(single_sample)
            #     pred_prob = H_DT_classifier.predict_proba(single_sample)
            # else:
            #     prediction = H_KNN_classifier.predict(single_sample)
            #     pred_prob = H_KNN_classifier.predict_proba(single_sample)

            if prediction == 0:
                st.text("")
                st.warning("Customer won't need a housing loan")
                H_pred_probability_score = {"Won't need housing loan":pred_prob[0][0]*100,"May need housing loan":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.info(H_pred_probability_score)
                	
							
            else:
                st.text("")
                st.success("Customer may need a housing loan")
                H_pred_probability_scoreY = {"Won't need housing loan":pred_prob[0][0]*100,"May need housing loan":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.json(H_pred_probability_scoreY)

    elif choice == 'Predict Personal Loan Necessity':
        st.info("Prediction if Customer will need a Personal Loan")
        education = st.selectbox("Education", tuple(edu_dict.keys()))
        job   = st.selectbox("Occupation", tuple(job_dict.keys()) )
        month	 = st.selectbox("Whiich month was the client last contacted in? ",tuple(month_dict.keys()) ) 
        default	 = st.radio("Has a credit balance in a loan?", tuple(feature_dict.keys()) )
        balance = st.text_input("Savings account balance:")
        ifCreatedAccount	 = st.radio("Has a Fixed Deposit Account?", tuple(feature_dict.keys()) )

        feature_list = [get_value(education, edu_dict), get_value(job,job_dict), get_fvalue(default), get_value(month, month_dict),balance,get_fvalue(ifCreatedAccount)]
        single_sample = np.array(feature_list).reshape(1,-1)


        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        st.text("")
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
                prediction = P_RF_classifier .predict(single_sample)
                pred_prob = P_RF_classifier .predict_proba(single_sample)
            # elif model_choice == "Decision Tree Classifier":
            #     prediction = P_DT_classifier.predict(single_sample)
            #     pred_prob = P_DT_classifier.predict_proba(single_sample)
            # else:
            #     prediction = P_KNN_classifier.predict(single_sample)
            #     pred_prob = P_KNN_classifier.predict_proba(single_sample)

            if prediction == 0:
                st.text("")
                st.warning("Customer won't need a personal loan")
                L_pred_probability_score = {"Won't need personal loan":pred_prob[0][0]*100,"May need personal loan":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.info(L_pred_probability_score)
                	
							
            else:
                st.text("")
                st.success("Customer may need a housing loan")
                L_pred_probability_scoreY = {"Won't need personal loan":pred_prob[0][0]*100,"May need personal loan":pred_prob[0][1]*100}
                #st.markdown(result_temp,unsafe_allow_html=True)
                st.text("")
                st.subheader("Prediction Probability Score using {}".format(model_choice))
                st.json(L_pred_probability_scoreY)    				


    elif choice == 'Home':
       # st.markdown("<h1 style='text-align: center; color: black; font-size: 60px'>Bank Data Analysis 🏦</h1>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; color: black; font-size: 20px'>This application makes 4 different types of predictions using the Bank Marketing Dataset.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: black; font-size: 20px'>The predictions are: </p>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: black; font-size: 20px'>1. Predicts if a customer creates a Fixed Deposit account or not </h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: black; font-size: 20px'>2. Predicts if a customer is suitable to give a Bank Loan </h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: black; font-size: 20px'>3. Predicts if a customer may take a Housing Loan </h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: black; font-size: 20px'>4. Predicts if a customer may take a Personal Loan </h5>", unsafe_allow_html=True)
        

        @st.cache(allow_output_mutation=True)
        def get_base64_of_bin_file(bin_file):
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()

        def set_png_as_page_bg(jpg_file):
            bin_str = get_base64_of_bin_file(jpg_file)
            page_bg_img = '''
            <style>
            body {
            background-image: url("data:uu/jpg;base64,%s");
            background-size: cover;
            }
            </style>
            ''' % bin_str
            
            st.markdown(page_bg_img, unsafe_allow_html=True)
            return

        set_png_as_page_bg('ii4.jpg')


    else:
        st.info("Visualize Data")

        Data_Url =("bank-full-dataset.csv")
        @st.cache(persist=True)
        def load_data():
            data  =pd.read_csv(Data_Url)
            data['day'] = pd.to_datetime(data['day'])
            return data

        data = load_data()
        # st.write(data)

        st.sidebar.subheader("Bank Markerting Data Analysis")

        #row = st.sidebar.radio('Marital', ('married','single','divorced'))
        #st.sidebar.markdown(data.query('marital == @row')[["text"]].sample(n=1).iat[0,0])


        st.sidebar.markdown("Number Of People By Marital Status 🤵 | 👫 ")
        select = st.sidebar.selectbox('Visualization type', ['histogram','pie chart'],key='1')


        sentiment_count = data['marital'].value_counts()
        #st.write(sentiment_count)
        sentiment_count = pd.DataFrame({'Marital':sentiment_count.index, 'Count':sentiment_count.values})

        
        st.title("Number of customers having a housing loans")
        housing_count = data['housing'].value_counts()
    
        housing_count = pd.DataFrame({'housing':housing_count.index, 'Count':housing_count.values})
        plot3 = px.pie(housing_count,values='Count',names='housing')
        st.plotly_chart(plot3)


        st.title("Number of customers according to their occupation")
        job_count = data['job'].value_counts()
        job_count = pd.DataFrame({'job':job_count.index, 'Count':job_count.values})
        
        plot2 = px.bar(job_count, x='job' , y='Count',color='Count',height=500)
        st.plotly_chart(plot2)

        st.title("Number of customers according to education levels")
        edu_count = data['education'].value_counts()
    
        edu_count = pd.DataFrame({'education':edu_count.index, 'Count':edu_count.values})
        plot4 = px.pie(edu_count,values='Count',names='education')
        st.plotly_chart(plot4)

        st.title("Number Of customers according to Marital Status ")
        if select =="histogram":
            fig = px.bar(sentiment_count, x='Marital' , y='Count',color='Count',height=500)
            st.plotly_chart(fig) 

        else:
            fig =px.pie(sentiment_count,values='Count',names='Marital')
            st.plotly_chart(fig)


        st.sidebar.markdown( "Analysing if customer subscribed to Fixed Deposit or not ")
        select3 = st.sidebar.selectbox('Visualization type', ['histogram','pie chart'],key='2')


        sentiment_count3 = data['y'].value_counts()

        sentiment_count3 = pd.DataFrame({'Bank term deposit':sentiment_count3.index, 'Count':sentiment_count3.values})



        st.title("Analysing if customer subscribed to Fixed Deposit or not")
        if select3 =="histogram":
            fig = px.bar(sentiment_count3, x='Bank term deposit' , y='Count',color='Count',height=500)
            st.plotly_chart(fig)

        else:
            fig =px.pie(sentiment_count3,values='Count',names='Bank term deposit')
            st.plotly_chart(fig)




        st.title("Total Call Durations By Month")



        #selected_metrics = st.selectbox(
            #label="Choose...", options=['Duration','Campaign','Recoveries']
        #)

        # Create traces
        fig = go.Figure()
        #if selected_metrics == 'Duration':
        fig.add_trace(go.Scatter(x=data.month, y=data.duration,
                        mode='markers',
                        name='duration'))


        st.plotly_chart(fig, use_container_width=True)




        st.title("Total Campaigns By Month")

        fig1 = go.Figure()
        #if selected_metrics == 'Campaign':
        fig1.add_trace(go.Scatter(x=data.month, y=data.campaign,
                            mode='markers', name='campaign'))

        st.plotly_chart(fig1, use_container_width=True)


    

if __name__ == '__main__':
	main()