import sklearn
from sklearn.cluster import KMeans
import xgboost as xg
import numpy as np 
import pandas as pd
import pickle
import sbpkg as sb
import clbg as cl
from flask import Flask, request, make_response
import numpy as np 
import pandas as pd
import pickle
import joblib
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)


#Loading model using pickle file 
kmean_model = open('kmeans_demographic_model.pkl','rb')

clustering = pickle.load(kmean_model)




@app.route('/',methods=['GET'])

def base_route():
    #return render_template("home.html")
     return"Marketing Performace Analysis on Recent Campaign Offers and predicting the cutomer spend "



@app.route('/predict',methods=['POSt'])
def predict():
        """Testing of prediction 
        ---
        parameters:  
             - name: customer_profile
               in: formData
               type: file
               required: true
             - name: customer_transaction
               in: formData
               type: file
               required: true 
             - name: customer_portfolio
               in: formData
               type: file
               required: true
        responses:
                200:
                    description: The response is 
                
        """

        df_1 = request.files['customer_profile']
        df1_pd = pd.read_json(df_1,lines=True)
        df_2 = request.files['customer_transaction']
        df2_pd = pd.read_json(df_2,lines=True)
        df_3 = request.files['customer_portfolio']
        df3_pd = pd.read_json(df_3,lines=True)
  

        # run the initial cleaning on each dataset
        clean_port_df = sb.clean_portfolio_data(df3_pd)
        clean_prof_df = cl.clean_profile_data(df1_pd)
        clean_trans_df = sb.clean_transcript_data(df2_pd)

        # calculates the uninfluenced transactions for the modeling
        uninflunced_trans = sb.norm_transactions(clean_trans_df, clean_port_df)

        # process the user data to create the modeling input
        user_data = sb.user_transactions(clean_prof_df, uninflunced_trans)

        # load in the user spend by day
        spd = sb.spend_per_day(clean_trans_df, clean_port_df)
        spd.reset_index(inplace=True)

        # predict the demographic of all the users in the data
        predictions = sb.predict_demographic(user_data)

        # drop everything except the demographic from the prediction data
        demographics = predictions[['person','demographic']]

        # merge the two datasets so that we have the demographic data for each person
        input_data = spd.merge(demographics, on=['person'])

        # only keep the first 23 days so that the last 7 can be used for modeling
        input_data = input_data[input_data['transaction_time'] < 24]

        # sum the spend & number of offers that 
        input_data = input_data.groupby(['transaction_time','person']).sum()
        input_final = input_data

        def create_dummy_days(data):

            # add dummies for the days of the week 
            
            day_of_week = pd.DataFrame(list(data['transaction_time']))
            for n in [1,2,3,4,5,6,7]:
                day_of_week = day_of_week.replace((n+7), n).replace((n+7*2), n).replace((n+7*3), n).replace((n+7*4), n)   
            day_of_week = pd.DataFrame([str(x) for x in day_of_week.iloc[:,0]])
            input_data_test = pd.concat([data, pd.get_dummies(day_of_week)], axis=1, join='inner')
            
            return input_data_test

        # reset the index
        input_final.reset_index(inplace=True)
        

        def predict_spend(input_data_test, model_demographic):
            
            # load in the model needed to predict spend on the analysis
            demo_model = joblib.load(f"xgboost_price_model_{model_demographic}.pkl")
            
            
            # keep only the demographic data related to the model to be used in this section
            data = input_data_test[input_data_test['demographic'] == model_demographic]
            
            # reset the index
            data.reset_index(inplace=True,drop=True)
            
            # add dummies for the days of the week 
            input_data_test_ = create_dummy_days(data)

            # keep only the columns needed
            input_data_test1 = input_data_test_[[ '0b1e1539f2cc45b7b9fa7c272da2e1d7', '2298d6c36e964ae4a3e7e9706d1fb8c2',
                                                '2906b810c7d4411798c6938adc9daaa5', '3f207df678b143eea3cee63160fa8bed',
                                                '4d5c57ea9a6940dd891ad53e9dbe8da0', '5a8bc65990b245e5a138643cd4eb9837',
                                                '9b98b8c7a33c4b65b9aebfe6a799e6d9', 'ae264e3637204a6fb9bb56bc8210ddfd',
                                                'f19421c1d4aa40978ebb69ca19b0e20d', 'fafdcd668e3743c1bb461111dcafc2a4','0_1.0', '0_2.0', '0_3.0', '0_4.0', '0_5.0', '0_6.0','0_7.0']]
            input_data_test1 = input_data_test1.drop_duplicates(keep="first")
            print(input_data_test1)
            # calculate the prediction based on the input date
            prediction = demo_model.predict(input_data_test1)
            print(prediction)
            # attach the prediction to the original filtered df
            input_data_test1['Customer_Spend_prediction'] = prediction
            
            return input_data_test1

                # add dummies for the days of the week 
        

        # keep only the columns needed
        #input_data_test_ = input_data_test_.drop(columns=['transaction_time','person','demographic'],inplace=True,axis=1)
        
        # create a function to use each of the above models to predict spend
        
        final=[]
        for demographic in [0,1,2,3,4,5,6,7]:
            final_res=predict_spend(input_final, demographic)
            final.append(final_res)

        return f"The customer spend is  {final}"
        

if __name__ == "__main__":
    app.run()