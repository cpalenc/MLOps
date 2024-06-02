import pandas as pd
from scipy.stats import entropy, ks_2samp

def kl_divergence(p, q):
    return entropy(p, q)

def test_kulball(df1, df2):

    threshold_kl = 0.1  # Define un umbral apropiado para tu contexto
    n = df1.shape[1]
    t = 0

    for columns in df1.columns:
                
        historical_feature = df1[columns].value_counts(normalize=True)
        recent_feature = df2[columns].value_counts(normalize=True)

        all_index = historical_feature.index.union(recent_feature.index)
        p = historical_feature.reindex(all_index, fill_value=0)
        q = recent_feature.reindex(all_index, fill_value=0)

        kl_div = kl_divergence(p, q)

        if kl_div > threshold_kl:
            t = t + 1
            # print(columns)
        else:
            t = t
            # print(columns)

    s = t / n

    return s

def test_shapiro(df1, df2):

    threshold_p_value = 0.05  # Nivel de significancia
    n = df1.shape[1]
    t = 0

    for columns in df1.columns:
                
        ks_stat, p_value = ks_2samp(df1[columns], df2[columns])

        if p_value < threshold_p_value:
            t = t + 1
            # print(columns)
        else:
            t = t
            # print(columns)

    s = t / n

    return s


def testeo_general(df1, df2):
    v1 = test_kulball(df1, df2)
    v2 = test_shapiro(df1, df2)

    vf = (v1+v2)/2
    
    if(df2.shape[0] > df1.shape[0]):
        n = df1.shape[0]/df2.shape[0]
    else:
        n = df2.shape[0]/df1.shape[0]

    print(vf, n)

    if (vf > 0.6) & (n>= 0.1):
        return 't1'
    else:    
        return 't2'



def clean_tester(df_test):

    

    df_test = df_test.dropna()
    df_test["año"] = pd.to_datetime(df_test['prev_sold_date']).dt.year
    df_test["decada"] = (df_test["año"] // 10) * 10

    df_test = df_test[df_test['bed'] < 7]
    df_test = df_test[df_test['bath'] < 5]
    df_test = df_test[df_test['price'] < 300000]
    df_test = df_test[df_test['acre_lot'] <= 0.0894211]
    df_test = df_test[df_test['house_size'] < 3500]
    df_test = df_test[df_test['decada'] >= 1980]
    
    state_dict = {
            'alabama': 1, 'alaska': 2, 'arizona': 3, 'arkansas': 4, 'california': 5, 'colorado': 6,
            'connecticut': 7, 'delaware': 8, 'florida': 9, 'georgia': 10, 'hawaii': 11, 'idaho': 12,
            'illinois': 13, 'indiana': 14, 'iowa': 15, 'kansas': 16, 'kentucky': 17, 'louisiana': 18,
            'maine': 19, 'maryland': 20, 'massachusetts': 21, 'michigan': 22, 'minnesota': 23,
            'mississippi': 24, 'missouri': 25, 'montana': 26, 'nebraska': 27, 'nevada': 28,
            'nueva hampshire': 29, 'nueva jersey': 30, 'nueva york': 31, 'nuevo mexico': 32,
            'carolina del norte': 33, 'dakota del norte': 34, 'ohio': 35, 'oklahoma': 36, 'oregon': 37,
            'pensilvania': 38, 'rhode island': 39, 'carolina del sur': 40, 'dakota del sur': 41,
            'tennessee': 42, 'texas': 43, 'utah': 44, 'vermont': 45, 'virginia': 46, 'washington': 47,
            'virginia occidental': 48, 'wisconsin': 49, 'wyoming': 50
                }
    
    df_test['state'] = df_test['state'].str.lower()
    df_test['state'] = df_test['state'].map(state_dict)
   
    df_test = df_test.loc[:,['price','bed','bath','acre_lot','state','house_size']]
    df_test = df_test.dropna(how = 'all')



    return df_test