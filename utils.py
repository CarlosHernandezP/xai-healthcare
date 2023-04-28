from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
import pandas as pd
import numpy as np
import re

def pre_process_seer(df : pd.DataFrame) -> pd.DataFrame:
    categorical_cols = []
    categories = []

    ### Age - TODO divide into three total groups
    df['Age_int'] = [int(x[:2]) for x in df['Age recode with single ages and 90+']]
    df['Age_coded'] = [2 if x>=65 else 1 if (x<65) and (x>=45) else 0 for x in df['Age_int']]

    ### Tumor size
    df['Tumor_size_coded'] = df['CS tumor size (2004-2015)'].apply(lambda x: np.log(x+10e-6))

    ### Tumor extension -> Coded as OHE |||| MAYBE try it out as continuous variable
    tumor_ext_values = [100, 330, 400, 200, 300, 310, 500, 315, 350, 355, 375, 320, 340,
           360, 380, 800, 335, 370, 950]

    df['CS extension (2004-2015)_coded'] = df['CS extension (2004-2015)']
    #tumor_ext_values_str = [str(x) for x in tumor_ext_values]
    #df['CS extension (2004-2015)'] = df['CS extension (2004-2015)'].astype(str)
    #categories.append(tumor_ext_values_str)
    #categorical_cols.append('CS extension (2004-2015)')

    ### RX Summ--Surg Prim Site (1998+) -> convert to int
    df['Prim_site_coded'] = df['RX Summ--Surg Prim Site (1998+)'].astype(int)

    ### Sex binary variable
    df['Sex_coded'] = [1 if x=='Female' else 0 for x in df['Sex']]

    ### Race -> Categorical variable
    # define the mapping of old to new values
    race_mapping = {
        'White': 'White',
        'Black': 'Other',
        'Other (American Indian/AK Native, Asian/Pacific Islander)': 'Other',
        'Unknown': 'Other'
    }

    df['Race_coded'] = [1 if x=='White' else 0 for x in df['Race recode (White, Black, Other)']]


    ### Marital status -> OHE TBD how to combine them
    marital_mapping = {
        'Married (including common law)': 'Married',
        'Widowed': 'Married',
        'Unmarried or Domestic Partner': 'Married',
        'Single (never married)': 'Single',
        'Unknown': 'Other',
        'Divorced': 'Other',
        'Separated': 'Other'
    }

    df['Marital_mapped'] = df['Marital status at diagnosis'].replace(marital_mapping)
    marital_values = ['Single', 'Married', 'Other']

    categories.append(marital_values)
    categorical_cols.append('Marital_mapped')

    ### Primary Site - labeled -> OHE with the classes being HF, trunk and limbs
    primary_site_mapping = { 
        'C44.0-Skin of lip, NOS': 'HF',
        'C44.1-Eyelid': 'HF',
        'C44.2-External ear': 'HF',
        'C44.3-Skin other/unspec parts of face': 'HF',
        'C44.4-Skin of scalp and neck': 'HF',

        'C44.5-Skin of trunk': 'Trunk',

        'C44.6-Skin of upper limb and shoulder': 'Limbs',
        'C44.7-Skin of lower limb and hip': 'Limbs',
        'C44.8-Overlapping lesion of skin': 'Limbs',
        'C44.9-Skin, NOS': 'Limbs'
    }


    df['Primary_Site_enc'] = df['Primary Site - labeled'].replace(primary_site_mapping)
    primary_site_values = ['HF', 'Limbs', 'Trunk']
    categories.append(primary_site_values)
    categorical_cols.append('Primary_Site_enc')

    # T
    t_values = ['T1a', 'T2a',  'T1NOS', 'T2b', 'T1b', 'T3b', 'T3a', 'TX', 'T4b',
           'T4a', 'T2NOS', 'T4NOS', 'T3NOS']
    # regular expression pattern to extract T1-T4 and TX
    pattern = r'^T[0-4X]{1}'

    # extract the required values using regex
    df['Derived AJCC T, 6th ed (2004-2015)'] = [re.match(pattern, value).group(0) for value in df['Derived AJCC T, 6th ed (2004-2015)']]
    t_values = ['T0', 'T1', 'T2', 'T3', 'TX', 'T4']

    categories.append(t_values)
    categorical_cols.append('Derived AJCC T, 6th ed (2004-2015)')

    # N --- TODO N0 is 0 and the rest is 1
    n_map = {'N0' : 'N0', 'N1a': 'N1', 'N1b': 'N1', 'N1NOS': 'N1', 'N2a': 'N2', 'N2b': 'N2', 'N2c': 'N2', 'N2NOS': 'N2', 'NX' :'NX', 'N3' : 'N3'}
    df['N_enc'] = df['Derived AJCC N, 6th ed (2004-2015)'].replace(n_map)

    n_values = df['N_enc'].unique()

    categories.append(n_values)
    categorical_cols.append('N_enc')
    #df['N_coded'] = [0 if x=='N0' else 1 for x in df['Derived AJCC N, 6th ed (2004-2015)']]


    # M TODO M0 is 0 and the rest is 1
    m_map = {
        'MX': 'MX',
        'M0': 'M0',
        'M1a': 'M1',
        'M1b': 'M1',
        'M1c': 'M1',
        'M1NOS': 'M1'
    }

    df['M_enc'] = df['Derived AJCC M, 6th ed (2004-2015)'].replace(m_map)

    m_values = df['M_enc'].unique()

    categories.append(m_values)
    categorical_cols.append('M_enc')


    ### Summary stage 2000 (1998-2017) -> Ordinal variable or OHE
    #summary_values = df['Summary stage 2000 (1998-2017)'].unique()

    #categories.append(summary_values)
    #categorical_cols.append('Summary stage 2000 (1998-2017)')
    df['Summary_stage_coded'] = [0 if x=='Localized' else 1 if x=='Regional' else 2 for x in df['Summary stage 2000 (1998-2017)']]

    ### Radiation recode -> binary variable
    # create a dictionary of values to be replaced with 1

    set_as_0 = ['None/Unknown',  'Refused (1988+)', 'Recommended, unknown if administered']
    df['Radiation_coded'] = [0 if x in set_as_0 else 1 for x in df['Radiation recode']]

    ### Chemotherapy recode (yes, no/unk) -> Binary
    df['Chemotherapy_coded'] = [1 if x.lower()=='yes' else 0 for x in df['Chemotherapy recode (yes, no/unk)']]

    ### RX Summ--Scope Reg LN Sur (2003+) -> None and Unknown or not applicable as NO and the rest YES
    df['Scope_Reg_coded'] = [0 if x in ['None', 'Unknown or not applicable'] else 1 for x in df['RX Summ--Scope Reg LN Sur (2003+)']]

    ### RX Summ--Surg/Rad Seq -> No rad... as NO the rest YES
    df['Surg/Rad_coded'] = [0 if x=='No radiation and/or cancer-directed surgery' else 1 for x in df['RX Summ--Surg/Rad Seq']]

    ### Median household income inflation adj to 2021 -> Two to three categories OHE
    # define a function to map income to categories
    def income_categories(income):
        if income.startswith('$75,000+'):
            return 2
        elif income.startswith('$65,000') or income.startswith('$70,000'):
            return 1
        else:
            return 0

    # apply the function to the income column and create a new column for the categories
    df['Income_coded'] = df['Median household income inflation adj to 2021'].apply(income_categories)


    # instantiate the ColumnTransformer object
    ct = ColumnTransformer([
            ("encoder", OneHotEncoder(categories=categories, drop='first'), categorical_cols)
        ], remainder="drop")

    all_but_first = [x[1:] for x in categories]
    flat_list = [element for sublist in all_but_first for element in sublist]

    # fit the ColumnTransformer to the data and transform it
    # fit the ColumnTransformer to the data and transform it
    transformed_data = ct.fit_transform(df)
    transformed_df = pd.DataFrame(data=transformed_data.toarray(), columns=flat_list)

    # assign transformed columns to original DataFrame
    df.loc[:, transformed_df.columns] = transformed_df

    # get columns that end with '_coded'
    coded_cols = [col for col in df.columns if col.endswith('_coded')]

    # get columns from categorical_variables list
    for sublist in categories:
        for col in sublist:
            if col not in coded_cols and col not in ['Single', 'T0', 'HF', '100', 'M0', 'N0']:
                coded_cols.append(col)


    df_train = df[coded_cols]


    # create a StandardScaler object
    scaler = StandardScaler()

    # fit the scaler to the data and transform it
    scaled_data = scaler.fit_transform(df_train)

    # convert the numpy array back to a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=df_train.columns)
    
    return scaled_df