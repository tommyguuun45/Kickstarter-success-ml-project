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
    'converted_pledged_amount','backers_count','state_changed_at', 'created_at'], axis=1)
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
    df[["goal"]]=df[["goal"]].round(0)
    return df

def make_encode(df:pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop = True)
    df = pd.get_dummies(df, columns=["country", "currency", "current_currency", "staff_pick", "usd_type","category_slug"], drop_first=True)
    return df

def check_column_completeness(df:pd.DataFrame) -> pd.DataFrame:
    column_list = ['deadline', 'goal', 'launched_at', 'state', 'country_AT',
       'country_AU', 'country_BE', 'country_CA', 'country_CH', 'country_DE',
       'country_DK', 'country_ES', 'country_FR', 'country_GB', 'country_HK',
       'country_IE', 'country_IT', 'country_JP', 'country_LU', 'country_MX',
       'country_NL', 'country_NO', 'country_NZ', 'country_SE', 'country_SG',
       'country_US', 'currency_AUD', 'currency_CAD', 'currency_CHF',
       'currency_DKK', 'currency_EUR', 'currency_GBP', 'currency_HKD',
       'currency_JPY', 'currency_MXN', 'currency_NOK', 'currency_NZD',
       'currency_SEK', 'currency_SGD', 'currency_USD', 'current_currency_CAD',
       'current_currency_USD', 'staff_pick_False', 'staff_pick_True',
       'usd_type_domestic', 'usd_type_international', 'category_slug_art',
       'category_slug_comics', 'category_slug_crafts', 'category_slug_dance',
       'category_slug_design', 'category_slug_fashion',
       'category_slug_film & video', 'category_slug_food',
       'category_slug_games', 'category_slug_journalism',
       'category_slug_music', 'category_slug_photography',
       'category_slug_publishing', 'category_slug_technology',
       'category_slug_theater']
    for column in column_list:
        if column in df.columns:
            continue
        else:
            df.insert(column_list.index(column), column, 0)
    return df


#We combine all the functions from above into one function in order to increase code clarity in the other py.-files
def preprocessing(df:pd.DataFrame)-> pd.DataFrame:
    df = extract_dict_item(df)
    df = drop_column(df)
    df = filter_transform_target(df)
    df = round_values(df)
    df = make_encode(df)
    df = check_column_completeness(df)
    return df
    