# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import data_preprocess
import predict_model
from sklearn.model_selection import train_test_split
import os
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    folder_name = 'train_test_4part'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # # --------換要上傳的測資時要刪掉(這是在平台給的train中自己切train test來建模)-----
    # data = pd.read_csv("30_Training Dataset_V2/training_data.csv")
    # split_index = int(0.8 * len(data))
    # train_data = data[:split_index]
    # test_data = data[split_index:]
    # train_data = train_data.reset_index(drop=True)
    # test_data = test_data.reset_index(drop=True)
    
    # train_data_list, test_data_list = data_preprocess.CutIntoFourParts(train_data)
    # test_data_list = data_preprocess.CutIntoFourParts(test_data)
    # y_test_list = []
    # for i in range(4):
    #     y_test_list.append(test_data_list[i]['單價'])
    #     test_data_list[i].drop('單價', axis=1, inplace=True)
    # # --------------------------------------------------
    train_data = pd.read_csv("30_Training Dataset_V2/training_data.csv")
    test_data = pd.read_csv("30_Public Dataset_Public Sumission Template_v2/public_dataset.csv")
    
    # 依據不同縣市資料多寡，將資料區分為四個part用不同模型預測，詳細區分依據見function宣告處
    train_data_list = data_preprocess.CutIntoFourParts(train_data)
    test_data_list = data_preprocess.CutIntoFourParts(test_data)
    
    all_result_df = pd.DataFrame()
    for i in range(4):
    
        # '縣市'跟'鄉鎮市區'合併
        train_data_list[i]['縣市區'] = train_data_list[i]['縣市'] + train_data_list[i]['鄉鎮市區']
        test_data_list[i]['縣市區'] = test_data_list[i]['縣市'] + test_data_list[i]['鄉鎮市區']
            
        # 切成三段
        train_data_list[i]['移轉層次'], test_data_list[i]['移轉層次'] = data_preprocess.floor_processing(train_data_list[i]['移轉層次'], test_data_list[i]['移轉層次'])
        
        # 不會用到的features drop掉
        train_data_list[i].drop(['備註', '車位個數', '總樓層數', '使用分區', '縣市', '鄉鎮市區',
                        '路名', "土地面積", "建物面積", "主建物面積"], axis=1, inplace=True)
        test_data_list[i].drop(['備註', '車位個數', '總樓層數', '使用分區', '縣市', '鄉鎮市區',
                        "土地面積", "建物面積", "主建物面積", '路名'], axis=1, inplace=True)

        # outlier detect
        train_data_list[i] = data_preprocess.outlier_detect(train_data_list[i])
        test_data_list[i] = data_preprocess.outlier_detect(test_data_list[i])

        # 處理座標 - TWD97轉十進位
        train_data_list[i][['橫坐標', '縱坐標']] = data_preprocess.coordinate_processing(train_data_list[i][['橫坐標', '縱坐標']])
        test_data_list[i][['橫坐標', '縱坐標']] = data_preprocess.coordinate_processing(test_data_list[i][['橫坐標', '縱坐標']])

    # ------------------------------
        train_data_list[i].to_excel((os.path.join(folder_name,
                                                "ori_train_data_{}.xlsx".format(i))), index=False)
        test_data_list[i].to_excel((os.path.join(folder_name,
                                                "ori_test_data_{}.xlsx".format(i))), index=False)
        # train_data_list[i] = pd.read_excel("train_test_4part/ori_train_data_{}.xlsx".format(i), engine='openpyxl')
        # test_data_list[i] = pd.read_excel("train_test_4part/ori_test_data_{}.xlsx".format(i), engine='openpyxl')
    # ------------------------------
        # 處理座標 - 合併額外資料的座標(方圓幾公里內有沒有某設施)
        train_coordinate = data_preprocess.combine_external_coordinate(train_data_list[i][['橫坐標', '縱坐標']])
        test_coordinate = data_preprocess.combine_external_coordinate(test_data_list[i][['橫坐標', '縱坐標']])
        train_data_list[i] = pd.concat([train_coordinate, train_data_list[i]], axis=1)
        test_data_list[i] = pd.concat([test_coordinate, test_data_list[i]], axis=1)
        
        # train_data_list[i].drop(['橫坐標', '縱坐標'], axis=1, inplace=True)
        # test_data_list[i].drop(['橫坐標', '縱坐標'], axis=1, inplace=True)

        # test_data"ID"先拿掉
        X_test_ID = test_data_list[i]['ID']
        test_data_list[i].drop('ID', axis=1, inplace=True)

        train_continue_data = train_data_list[i][['屋齡', '車位面積', '陽台面積', '附屬建物面積', '橫坐標', '縱坐標']]
        y_train = train_data_list[i]['單價'] # 獲得label
        train_categorical_data = train_data_list[i].drop(['屋齡', '車位面積', '陽台面積', '附屬建物面積', '橫坐標', '縱坐標', '單價', 'ID'], axis=1)

        test_continue_data = test_data_list[i][['屋齡', '車位面積', '陽台面積', '附屬建物面積', '橫坐標', '縱坐標']]
        test_categorical_data = test_data_list[i].drop(test_continue_data.columns, axis=1)

        # OneHotEncoding
        train_categorical_data , test_categorical_data = data_preprocess.OneHotEncoding(train_categorical_data , test_categorical_data)

        # Normalization
        train_continue_data, test_continue_data = data_preprocess.MinMaxNormalization(train_continue_data, test_continue_data)

        # combine continue_data & categorical_data
        X_train = pd.concat([train_continue_data, train_categorical_data], axis=1)
        X_test = pd.concat([test_continue_data, test_categorical_data], axis=1)

        # RandomForest_FeatureSelect 11.多
        # X_train = data_preprocess.RandomForest_FeatureSelect(X_train, y_train)

        # 因為OneHotEncoding後，有些train裡面有的column，test裡面沒有，模型預測上會特徵數量對不上
        # 所以如果有test_categorical_data沒有的column就把該column刪掉
        X_train , X_test = data_preprocess.DropSingleColumn(X_train , X_test)

        # pca降維 12.多
        # n_Dimension = 50
        # X_train, X_test = data_preprocess.PCA_DimensionalityReduction(X_train, X_test, n_Dimension)
        
        # smote_augmentation
        # X_train, y_train = predict_model.smote_augmentation(X_train, y_train)

        # shuffle
        X_train, y_train = data_preprocess.shuffle(X_train, y_train)

    # ------------------------------------------------
        if i == 0: # 新北、台北的用mlp_model
            model = predict_model.mlp_model(X_train.shape[1])
            model.fit(X_train, y_train, epochs=90, batch_size=32, validation_split=0.2)

            y_pred = model.predict(X_test)
            result_df = pd.DataFrame({'ID': X_test_ID, 'predicted_price': y_pred.flatten()})
            
            # mape_value = predict_model.calculate_mape(y_test_list[i], y_pred)
            # print("model_{} - MAPE:".format(i), mape_value)
    # ------------------------------------------------
        else:  
            # model_RandomForest
            model = predict_model.model_RandomForest()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            result_df = pd.DataFrame({'ID': X_test_ID, 'predicted_price': y_pred.flatten()})
            
            # mape_value = predict_model.calculate_mape(y_test_list[i], y_pred)
            # print("model_{} - MAPE:".format(i), mape_value)
        
        # result
        all_result_df = pd.concat([all_result_df, result_df], axis=0)
    
    all_result_df.to_csv("submission.csv", index=False)