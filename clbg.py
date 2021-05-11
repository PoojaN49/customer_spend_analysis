"""
---
Contains all of the useful functions needed to clean and process the Starbucks data ready for modeling
---
|_
|_ 
|_ 

---
"""
# import general functions
import pandas as pd
import numpy as np
import json

# modules for the modeling
import joblib
from sklearn.preprocessing import StandardScaler


def clean_profile_data(data):
    """
    this process clean gender, age and became_member_on columns in the profile data
    """
    # rename the column 'id' to person
    data.columns = ['gender','age','person' ,'member joined','income']
    
    # replace 118 in the age column with a zero indicating no age 
    # keeping these users as a seperate group is important as they may show different user behaviour
    data['age'] = data['age'].replace(118,0)

    # update the became_member_on column to a datetime format
    data['member joined'] = pd.to_datetime(data['member joined'], format='%Y%m%d')
    
    # replace the NaN's in the income
    data['income'] = data['income'].fillna(0)
    
    # replace M, F, O and None types to get the 4 groups of customers
    data['gender'] = data['gender'].replace('M','male')
    data['gender'] = data['gender'].replace('F','female')
    data['gender'] = data['gender'].replace('O','other')
    data['gender'] = data['gender'].fillna('unknown gender')

    # split the column into seperate columns
    temp_df = pd.get_dummies(data['gender'])

    # combine the dataframes
    data = pd.concat([temp_df, data], axis=1, sort=True)

    # drop the original column
    data = data.drop(columns=['gender'])

    return data