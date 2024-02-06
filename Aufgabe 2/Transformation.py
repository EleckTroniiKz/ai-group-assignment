import pandas as pd
columns_to_keep = [0, 2, 3, 4, 5]
data = pd.read_csv("Sample_Tracking_Data_for_Recommender_System_ABCD.csv", delimiter=';', index_col=0,usecols=columns_to_keep)

data = data[data["ACTION"]=="VIEW_ARTWORK"]
columns_with_nan = data.columns[data.isna().any()]
data_nan_timediff = data[pd.isna(data["TIMEDIFF_SECS"])]
# Gib die Namen der betroffenen Spalten aus
data = data.dropna(subset=['TIMEDIFF_SECS'])
data['ARTWORK_ID'] = data['ARTWORK_ID'].astype(int)
name_mapping = {'ALICE': 0, 'BOB': 1, 'CAROL': 2, 'DAN': 3}
data['USERNAME'] = data['USERNAME'].map(name_mapping)
grouped_data = data.groupby(['USERNAME', 'ARTWORK_ID'])['TIMEDIFF_SECS'].sum().reset_index()
print(grouped_data)
pivot_data = pd.pivot_table(grouped_data, values='TIMEDIFF_SECS', index='USERNAME', columns='ARTWORK_ID', aggfunc='sum')
pivot_data = pivot_data.fillna(0)
print(pivot_data)
#pivot_data = pivot_data.fillna(NAN)
pivot_data.to_csv("User-Item_Interaction_Matrix.csv", sep=';')