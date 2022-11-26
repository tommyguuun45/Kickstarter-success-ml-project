import pandas as pd
import numpy as np
import ast
import pickle



def extract_dict_item(df:pd.DataFrame)-> pd.DataFrame:
    '''
    This function take the panda DataFrame, extracts from the category column the first value of slug.
    Note that the values of the category column are dictionaries in a string.
    '''
    df['category'] = df['category'].apply(lambda x: ast.literal_eval(x))
    df['category_slug'] = df['category'].apply(lambda x: x.get('slug'))
    df['category_slug'] = df['category_slug'].apply(lambda x: x.split("/")[0])
    return df
    

# here we delete the unwanted columns!
def drop_column(df:pd.DataFrame) -> pd.DataFrame:
    df= df.drop(['blurb', 'category', 'creator', 'currency_symbol',
       'currency_trailing_code', 'friends', 'fx_rate', 'id',
       'is_backing', 'is_starred', 'location',
       'name', 'permissions', 'photo', 'pledged', 'profile', 'slug',
       'source_url', 'spotlight',
       'static_usd_rate', 'urls',"disable_communication", "is_starrable",'usd_pledged',
       'converted_pledged_amount','backers_count','state_changed_at'], axis=1)
    return df

def filter_transform_target(df:pd.DataFrame) -> pd.DataFrame:
    '''
    This function filter successful and failed in the target (state column in dataframe).
    It replace them with 0 and 1 respectively and return the panda's DataFrame.
    '''
    df = df[(df.state != "live")] 
    df = df[(df.state != "canceled")] 
    df = df[(df.state != "suspended")] 
    df["state"] = df["state"].replace(["successful", "failed"], [1, 0])
    return df

def round_values(df:pd.DataFrame) -> pd.DataFrame:
    df[["goal","usd_pledged"]]=df[["goal","usd_pledged"]].round(0)
    return df

def make_encode(df:pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop = True)
    df = pd.get_dummies(df, columns=["country", "currency", "current_currency", "staff_pick", "usd_type","category_slug"])
    return df
    