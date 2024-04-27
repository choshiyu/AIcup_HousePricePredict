from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

# def mlp_model(dim):

#     model = Sequential()
#     model.add(Dense(256, input_dim=dim, activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='linear'))
    
#     model.compile(loss='mean_squared_error', optimizer='adam')

#     return model

def mlp_model(dim):
    
    model = Sequential()
    model.add(Dense(512, input_dim=dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

def model_RandomForest():

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    return model

def calculate_mape(y_true, y_pred):

    assert len(y_true) == len(y_pred), "len(y_true) != len(y_pred).....QQ"
    total_percentage_error = 0
    n = len(y_true)
    
    for i in range(n):
        true_value = y_true[i]
        pred_value = y_pred[i]
        
        # 計算百分比誤差
        percentage_error = abs((true_value - pred_value) / true_value) * 100
        total_percentage_error += percentage_error
    
    # MAPE
    mape = total_percentage_error / n
    
    return mape

def smote_augmentation(X_train, y_train):
    
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    return X_train, y_train