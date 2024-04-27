import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pyproj import Proj, transform
import math
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

def CutIntoFourParts(data):
    '''
    a"新北市","台北市" ----> 數量爆炸多，可用mlp試看看
    b"高雄市","桃園市","台中市","台南市"
    c"新竹縣","新竹市"
    d"基隆市","宜蘭縣","屏東縣","苗栗縣","金門縣","嘉義市","彰化縣","花蓮縣","雲林縣","嘉義縣"
    b、c、d依據城市的性質(?)來區分看看，數量少，用rf
    '''
    a_data = data[data['縣市'].isin(["新北市","台北市"])]
    a_data = a_data.reset_index(drop=True)
    b_data = data[data['縣市'].isin(["高雄市","桃園市","台中市","台南市"])]
    b_data = b_data.reset_index(drop=True)
    c_data = data[data['縣市'].isin(["新竹縣","新竹市"])]
    c_data = c_data.reset_index(drop=True)
    d_data = data[data['縣市'].isin(["基隆市","宜蘭縣","屏東縣","苗栗縣","金門縣","嘉義市","彰化縣","花蓮縣","雲林縣","嘉義縣"])]
    d_data = d_data.reset_index(drop=True)
    
    return [a_data, b_data, c_data, d_data]

def outlier_detect(data):

    # '車位面積'>6的都-自己的1/2
    condition = data['車位面積'] > 6
    data.loc[condition, '車位面積'] /= 2
    condition = data['車位面積'] > 5
    data.loc[condition, '車位面積'] -= 2
    
    # '陽台面積'>6的都-自己的1/2
    condition = data['陽台面積'] > 6
    data.loc[condition, '陽台面積'] /= 2
    
    # '附屬建物面積'>5的都-自己的1/2
    condition = data['附屬建物面積'] > 5
    data.loc[condition, '附屬建物面積'] /= 2
    condition = data['附屬建物面積'] > 5
    data.loc[condition, '附屬建物面積'] == 5
    
    # 畫圖
    # for col in data.columns:
    
    #     plt.hist(data[col], bins='auto', edgecolor='black')
    #     plt.xlabel('number')
    #     plt.ylabel('frequency')
    #     plt.title(f'{col}\'s data distribution')
    #     plt.show()
    
    return data
 
def floor_processing(data1, data2):
    '''
    切>=15 --> 3, 7~14 --> 2, <7 --> 1
    '''
    data1[data1 < 7] = 1
    data1[(data1 < 15) & (data1 >= 7)] = 2
    data1[data1 >= 15] = 3

    data2[data2 < 7] = 1    
    data2[(data2 < 15) & (data2 >= 7)] = 2       
    data2[data2 >= 15] = 3
    
    return data1, data2

def twd97_to_decimal_coordinate(lat, lng):

    # 定義TWD97座標系統的EPSG編碼
    twd97 = Proj(proj="tmerc", lat_0=0, lon_0=121, k=0.9999, x_0=250000, y_0=0, ellps="GRS80")

    # 定義WGS 84座標系統的EPSG編碼
    wgs84 = Proj(proj="latlong", datum="WGS84")

    # 將TWD97座標轉換為經緯度座標
    lng, lat = transform(twd97, wgs84, lat, lng)

    return lat, lng

def coordinate_processing(data):
    
    for idx, (i, j) in enumerate(zip(data['橫坐標'], data['縱坐標'])):
        new_i, new_j = twd97_to_decimal_coordinate(i, j)
        print(new_i, new_j)
        data.loc[idx, '橫坐標'] = new_i
        data.loc[idx, '縱坐標'] = new_j

    return data

def haversine_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371.0
    distance = r * c
    
    return distance

def combine_external_coordinate(data):
    # 除了平台給的external data，爬一些負面因子的座標下來
    # 先讀進其他資料的座標-ex. 醫院lat, 醫院lng
    ATM_data = pd.read_csv("30_Training Dataset_V2/external_data/ATM資料.csv", index_col=False)
    ATM_data = ATM_data[["lat", "lng"]]
    university = pd.read_csv("30_Training Dataset_V2/external_data/大學基本資料.csv", index_col=False)
    university = university[["lat", "lng"]]
    bus = pd.read_csv("30_Training Dataset_V2/external_data/公車站點資料.csv", index_col=False)
    bus = bus[["lat", "lng"]]
    train = pd.read_csv("30_Training Dataset_V2/external_data/火車站點資料.csv", index_col=False)
    train = train[["lat", "lng"]]
    Financial_Institutions = pd.read_csv("30_Training Dataset_V2/external_data/金融機構基本資料.csv", index_col=False)
    Financial_Institutions = Financial_Institutions[["lat", "lng"]]
    ConvenienceStore = pd.read_csv("30_Training Dataset_V2/external_data/便利商店.csv", index_col=False)
    ConvenienceStore = ConvenienceStore[["lat", "lng"]]    
    HighSchool = pd.read_csv("30_Training Dataset_V2/external_data/高中基本資料.csv", index_col=False)
    HighSchool = HighSchool[["lat", "lng"]]
    ElementarySchool = pd.read_csv("30_Training Dataset_V2/external_data/國小基本資料.csv", index_col=False)
    ElementarySchool = ElementarySchool[["lat", "lng"]]
    JuniorHighSchool = pd.read_csv("30_Training Dataset_V2/external_data/國中基本資料.csv", index_col=False)
    JuniorHighSchool = JuniorHighSchool[["lat", "lng"]]
    MRT_data = pd.read_csv("30_Training Dataset_V2/external_data/捷運站點資料.csv", index_col=False)
    MRT_data = MRT_data[["lat", "lng"]]
    PostOffice = pd.read_csv("30_Training Dataset_V2/external_data/郵局據點資料.csv", index_col=False)
    PostOffice = PostOffice[["lat", "lng"]]
    bike = pd.read_csv("30_Training Dataset_V2/external_data/腳踏車站點資料.csv", index_col=False)
    bike = bike[["lat", "lng"]]
    hospital = pd.read_csv("30_Training Dataset_V2/external_data/醫療機構基本資料.csv", index_col=False)
    hospital = hospital[["lat", "lng"]]
    garbage = pd.read_excel("30_Training Dataset_V2/Nfactors/垃圾掩埋場.xlsx", index_col=False, engine='openpyxl')
    garbage = garbage[["lat", "lng"]]
    IncinerationPlant = pd.read_csv("30_Training Dataset_V2/Nfactors/焚化廠座標.csv", index_col=False)
    IncinerationPlant = IncinerationPlant[["lat", "lng"]]
    gas_station = pd.read_excel("30_Training Dataset_V2/Nfactors/加油站.xlsx", index_col=False, engine='openpyxl')
    gas_station = gas_station[["lat", "lng"]]
    grave = pd.read_excel("30_Training Dataset_V2/Nfactors/墳墓.xlsx", index_col=False, engine='openpyxl')
    grave = grave[["lat", "lng"]]
    airport = pd.read_excel("30_Training Dataset_V2/Nfactors/airport.xlsx", index_col=False, engine='openpyxl')
    airport = airport[["lat", "lng"]]

    near_position = [ATM_data, bus, ConvenienceStore, MRT_data, bike, garbage, gas_station, grave, airport] # distance<0.7:1km, else:0
    near_position_name = ["ATM_data", "bus", "ConvenienceStore", "MRT_data", "bike", "garbage","gas_station","grave", "airport"]
    mid_position = [train, Financial_Institutions, PostOffice, IncinerationPlant] # distance<2km:1, else:0
    mid_position_name = ["train", "Financial_Institutions", "PostOffice", "IncinerationPlant"]
    far_position = [university, HighSchool, ElementarySchool, JuniorHighSchool, hospital] # distance<4km:1, else:0 
    far_position_name = ["university", "HighSchool", "ElementarySchool", "JuniorHighSchool", "hospital"]

    data_new_col = pd.DataFrame()
    for near_pos, near_name in zip(near_position, near_position_name):
        print("near")
        near_pos_new_df = pd.DataFrame()
        for x1, y1 in zip(data['橫坐標'], data['縱坐標']):
            has = False
            for x2, y2 in zip(near_pos["lat"], near_pos["lng"]):
                distance = haversine_distance(x1, y1, x2, y2)
                if distance < 0.7:
                    has = True
                    break
            if has == True:
                new_data = {near_name: 1}
            else:
                new_data = {near_name: 0}
            near_pos_new_df = near_pos_new_df.append(new_data, ignore_index=True)
        data_new_col[near_name] = near_pos_new_df.squeeze()
        
    for mid_pos, mid_name in zip(mid_position, mid_position_name):
        print("mid")
        mid_pos_new_df = pd.DataFrame()
        for x1, y1 in zip(data['橫坐標'], data['縱坐標']):
            has = False
            for x2, y2 in zip(mid_pos["lat"], mid_pos["lng"]):
                distance = haversine_distance(x1, y1, x2, y2)
                if distance < 2:
                    has = True
                    break
            if has == True:
                new_data = {mid_name: 1}
            else:
                new_data = {mid_name: 0}
            mid_pos_new_df = mid_pos_new_df.append(new_data, ignore_index=True)
        data_new_col[mid_name] = mid_pos_new_df.squeeze()
        
    for far_pos, far_name in zip(far_position, far_position_name):
        print("far")
        far_pos_new_df = pd.DataFrame()
        for x1, y1 in zip(data['橫坐標'], data['縱坐標']):
            has = False
            for x2, y2 in zip(far_pos["lat"], far_pos["lng"]):
                distance = haversine_distance(x1, y1, x2, y2)
                if distance < 4:
                    has = True
                    break
            if has == True:
                new_data = {far_name: 1}
            else:
                new_data = {far_name: 0}
            far_pos_new_df = far_pos_new_df.append(new_data, ignore_index=True)
        data_new_col[far_name] = far_pos_new_df.squeeze()

    return data_new_col

def OneHotEncoding(data1, data2):

    data1 = pd.get_dummies(data1)
    data2 = pd.get_dummies(data2)
    # OneHotColumnsName = OneHotDone_data.columns

    return data1, data2

def DropSingleColumn(data1, data2):
    '''
    因為OneHotEncoding後，有些train裡面有的column，test裡面沒有，模型預測上會特徵數量對不上
    所以如果有test_categorical_data沒有的column就把該column刪掉
    '''
    for column in data1.columns:
        if column not in data2.columns:
            data1 = data1.drop(column, axis=1)

    for column in data2.columns:
        if column not in data1.columns:
            data2 = data2.drop(column, axis=1)

    return data1, data2

def MinMaxNormalization(data1, data2):
    
    scaler = MinMaxScaler()
    
    data1_coor = data1[['橫坐標', '縱坐標']]
    data1 = data1.drop(['橫坐標', '縱坐標'], axis=1)
    data2_coor = data2[['橫坐標', '縱坐標']]
    data2 = data2.drop(['橫坐標', '縱坐標'], axis=1)
    
    columns = data1.columns.values
    data1 = scaler.fit_transform(data1.astype(np.float64))
    data1 = pd.DataFrame(data1)
    data1.columns = columns
    data1 = pd.concat([data1, data1_coor], axis=1)
    
    columns = data2.columns.values
    data2 = scaler.fit_transform(data2.astype(np.float64))
    data2 = pd.DataFrame(data2)
    data2.columns = columns
    data2 = pd.concat([data2, data2_coor], axis=1)

    return data1, data2

def PCA_DimensionalityReduction(X_train, X_test, n_Dimension):

    merged_data = np.concatenate((X_train, X_test), axis=0)
    pca = PCA(n_components=n_Dimension)
    pca_result = pca.fit_transform(merged_data)
    X_train = pca_result[:len(X_train)]
    X_test = pca_result[len(X_train):]

    return X_train, X_test

def RandomForest_FeatureSelect(X_train, y_train):
    
    X_train_copy = X_train

    rf = RandomForestRegressor(n_estimators=100)
    importance = 0
    for _ in range(6):  # Repeat 200 times to get overall results
        rf.fit(X_train_copy, y_train)
        importance += rf.feature_importances_

    threshold = 0.008
    selected_features = X_train_copy.columns[importance >= threshold].tolist()
    print("len(selected_features):", len(selected_features))
    print(selected_features)

    selected_X_train = X_train[selected_features]
    selected_X_train = selected_X_train.reset_index(drop=True)

    return X_train

def shuffle(X_train, y_train):
    
    X_train = np.array(X_train)# 轉成np等做shuffle
    y_train = np.array(y_train)

    shuffle_idx = np.random.permutation(len(X_train)) # shuffle
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    return X_train, y_train