import pandas as pd

columns_to_keep = [0, 2, 3, 4, 5, 6]
data = pd.read_csv("Sample_Tracking_Data_for_Recommender_System_ABCD.csv", delimiter=';', index_col=0,
                   usecols=columns_to_keep)

data = data[data["ACTION"] == "VIEW_ARTWORK"]
columns_with_nan = data.columns[data.isna().any()]
data_nan_timediff = data[pd.isna(data["TIMEDIFF_SECS"])]

data = data.dropna(subset=['TIMEDIFF_SECS'])
data['ARTWORK_ID'] = data['ARTWORK_ID'].astype(int)
name_mapping = {'ALICE': 0, 'BOB': 1, 'CAROL': 2, 'DAN': 3}
data['USERNAME'] = data['USERNAME'].map(name_mapping)

# Transform to get the User-Item matrix
grouped_data = data.groupby(['USERNAME', 'ARTWORK_ID'])['TIMEDIFF_SECS'].sum().reset_index()

pivot_data = pd.pivot_table(grouped_data, values='TIMEDIFF_SECS', index='USERNAME', columns='ARTWORK_ID', aggfunc='sum')
pivot_data = pivot_data.fillna(0)
pivot_data.to_csv("User-Item_Interaction_Matrix.csv", sep=';')

# Getting the User-Item Information Flat for better processing in the profile context
data = data.loc[pd.notna(data['BESCHREIBUNG'])]
grouped_data = data.groupby(['USERNAME', 'ARTWORK_ID'])['TIMEDIFF_SECS'].sum().reset_index()
grouped_data.to_csv("User-Item_Interaction.csv", sep=";", index=False)

# Getting the Item Description Prepared
data.drop(columns=['ACTION', 'USERNAME', 'TIMEDIFF_SECS'], inplace=True)
data.reset_index(drop=True, inplace=True)
data = data.groupby(['ARTWORK_ID']).agg({'BESCHREIBUNG': 'first'}).reset_index()
data.to_csv("Item-Description.csv", sep=";", index=False)
