import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def calc_missing_values(df): 
    total = df.isnull().sum().sort_values(ascending=False)
    percent = ((df.isnull().sum().sort_values(ascending=False)/df.shape[0])*100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return(missing_data)

def load_sentence_embeddings(text_col_name):
    return pd.read_csv(f'./{text_col_name}_embeddings.csv', index_col=0) 

def get_sentence_embeddings(text_cols):
    embs = []
    for text_col_name in text_cols:
        assert text_col_name in ['facilities','description']
        embs.append( load_sentence_embeddings(text_col_name) )

    if len(embs)>0:
        return pd.concat(embs, axis=1)
    else:
        return []


def map_energyEfficiency(x):
    mapping = ['A_PLUS', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    if str(x) in mapping:
        return mapping.index(x)
    else:
        return np.nan
    

def limit_min_max(df, fname, lmin=None, lmax=None):
    if lmax is not None:
        df.loc[df[fname]>lmax,fname] = lmax
    if lmin is not None:
        df.loc[df[fname]<lmin,fname] = lmin


def correct_decimals(x):
    """Divide the input by 10^p if input is larger than 10^p for p>2"""
    exp_order = np.floor(np.log10(x))
    if exp_order>2: 
        return x/10**exp_order
    else:
        return x


def handle_target(df, target='totalRent'):
    """Preprocess target variable to eliminate outliers and be consistent with
    `baseRent + heatingCosts + serviceCharge`. Log() of target is calculated to
    make it's distribution closer to Normal. 

    Args:
        df (pd.DataFrame): Raw df
        target (str, optional): Defaults to 'totalRent'.

    Returns:
        tuple: processed df and target
    """    
    df['heatingCosts'] = df['heatingCosts'].apply(correct_decimals)
    df['serviceCharge'] = df['serviceCharge'].apply(correct_decimals)

    assumedRent = df['baseRent'] + df['serviceCharge'].fillna(0) + df['heatingCosts'].fillna(0)
    # fill na totalRents with assumedRent
    df.loc[df['totalRent'].isna(),'totalRent'] = assumedRent[df['totalRent'].isna()]
    # fill with assumedRent if totalRents are less than assumedRent
    df.loc[df['totalRent']<assumedRent,'totalRent'] = assumedRent[df['totalRent']<assumedRent]

    eps = 1e-2
    qmin, qmax = df[target].quantile([eps, 1-eps])
    # df = df.loc[(df[target]>qmin) & (df[target]<qmax)]
    df.drop(df[(df[target]<=qmin) | (df[target]>=qmax)].index, axis=0, inplace=True)

    return df, np.log(df[target])


def handle_firing_heating(df):
    fname = 'heatingType'

    mapping = {
        'gas':'gas_heating',
        'district_heating':'district_heating',
        'oil' : 'oil_heating',
        'electricity' : 'electric_heating',
    }

    df.loc[df[fname].isna(),fname] = df[df[fname].isna()]['firingTypes'].apply(lambda x : mapping.get(x,x))
    df.loc[df[fname].isna(),fname] = 'other'




def handle_specific_columns(df):

    limit_min_max(df, 'floor', lmin=-2, lmax=20)
    limit_min_max(df, 'lastRefurbish', lmin=1950, lmax=2023)
    limit_min_max(df, 'livingSpace', lmin=0, lmax=200)
    limit_min_max(df, 'noParkSpaces', lmax=100)
    limit_min_max(df, 'noRooms', lmin=0, lmax=10)
    limit_min_max(df, 'numberOfFloors', lmin=0, lmax=20)
    limit_min_max(df, 'picturecount', lmin=1, lmax=40)
    limit_min_max(df, 'thermalChar', lmax=1000)
    limit_min_max(df, 'yearConstructed', lmin=1000, lmax=2023)

    df['picturecount'] = np.log(df['picturecount'])
    # df['picturecount'] = np.log(df['picturecount'])

    fname = 'condition' # feature name
    df.loc[df[fname]=='ripe_for_demolition', fname] = 'other'
    df.loc[df[fname].isna(), fname] = 'other'

    # df['energyEfficiencyClass'] = df['energyEfficiencyClass'].apply(map_energyEfficiency)
    handle_firing_heating(df)
    
    df.loc[df['interiorQual'].isna(),'interiorQual'] = 'other'

    return df
                

def transform_features(df, categoricals_as='str', text_cols=[]):
    """Main method for feature preprocessing and transformations

    Args:
        df (pd.DataFrame): Raw dataframe
        categoricals_as (str, optional): Categorical columns processing. Either cast to strings or
            use integer's for each unique class. Defaults to 'str'. If anything else use int encoding
        text_cols (list, optional): _description_. Defaults to [].

    Returns:
        tuple: features df and labels
    """    

    df, target = handle_target(df, 'totalRent')
    df = handle_specific_columns(df)

    # numericals
    numericals = ['floor','lastRefurbish','livingSpace','noParkSpaces', 'thermalChar','yearConstructed',
                #   'pricetrend',
                  ]

    # binary
    binary = ['balcony','cellar','garden', 'hasKitchen','lift','newlyConst']
    for x in binary:
        df[x] = df[x].astype(float)
        
    # categoricals
    categoricals = ['condition',
                    'heatingType',
                    'geo_bln',
                    'geo_krs',
                    # 'geo_plz',
                    'interiorQual',
                    'petsAllowed',
                    'telekomTvOffer',
                    'typeOfFlat',
                    ]

    for x in categoricals:
        if categoricals_as=='str':
            df[x] = df[x].astype(str)
        else:
           df[x] = label_encoder.fit_transform(df[x].values)


    # text
    embs = get_sentence_embeddings(text_cols)
    if len(embs)>0:
        text_embeddings_columns = list(embs.columns)
        df = pd.concat([df,embs], axis=1)
    else:
        text_embeddings_columns = []    

    for colname in text_cols:
        df[colname] = df[colname].fillna('').astype(str)

    return df[categoricals+numericals+binary+text_cols], target, categoricals, text_embeddings_columns

