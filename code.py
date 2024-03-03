import pandas as pd

dataset = pd.read_csv('FINAL DATASET.csv')
X = dataset.iloc[:, 1:7].values
Y = dataset.iloc[:, 7:19].values

m1=[]
m2=[]
m3=[]
m4=[]
m5=[]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE = LabelEncoder()
onehotencoder = OneHotEncoder(categories ='auto')
X = onehotencoder.fit_transform(X).toarray()
X=pd.DataFrame(X)
X.drop(X.columns[[0,12,43,56,58,59]], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Y_train = sc.fit_transform(Y_train)
Y_test = sc.transform(Y_test)

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
for i in range(12):
    regressor.fit(X_train,Y_train[:,i])
    b_pred=regressor.predict(X_test)
    b_pred=pd.DataFrame(b_pred)
    z=mean_absolute_error(Y_test[:,i], b_pred)
    m1.append(z)
print(m1)


import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
for i in range(12):
    xg_reg.fit(X_train,Y_train[:,i])
    b_pred=xg_reg.predict(X_test)
    b_pred=pd.DataFrame(b_pred)
    z=mean_absolute_error(Y_test[:,i], b_pred)
    m2.append(z)
print(m2)

from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU 
from keras.activations import relu, sigmoid
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
def create_model(layers,activation):
    model=Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return model
model=KerasRegressor( build_fn=create_model, verbose=0)
model
layers=[[6],[6,4,2]]
activations=['sigmoid', 'relu']
param_grid=dict(layers=layers,activation=activations,batch_size=[10,25], epochs=[100])
grid=GridSearchCV(estimator=model, param_grid=param_grid)
for i in range(12):
    grid.fit(X_train,Y_train[:,i])
    b_pred=grid.predict(X_test)
    b_pred=pd.DataFrame(b_pred)
    z=mean_absolute_error(Y_test[:,i], b_pred)
    m3.append(z)
print(m3)



from sklearn.svm import SVR
regressor1=SVR(kernel='rbf')
for i in range(12):
    regressor1.fit(X_train,Y_train[:,i])
    b_pred=regressor1.predict(X_test)
    b_pred=pd.DataFrame(b_pred)
    z=mean_absolute_error(Y_test[:,i], b_pred)
    m4.append(z)
print(m4)


from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from keras.wrappers.scikit_learn import KerasRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import GridSearchCV
def create_model(layers,activation):
    model=Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return model
model=KerasRegressor( build_fn=create_model, verbose=0)
layers=[[6,4],[6,4,2]]
activations=['sigmoid', 'relu']
param_grid=dict(layers=layers,activation=activations,batch_size=[10,25], epochs=[100])

stack = StackingCVRegressor(regressors=( 
                            RandomForestRegressor(n_estimators=10,random_state=0),SVR(kernel='rbf'),xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10),GridSearchCV(estimator=model, param_grid=param_grid) ,SVR(kernel='rbf')),
meta_regressor=LinearRegression(), cv=8,
use_features_in_secondary=True,
store_train_meta_features=True,
shuffle=False,
random_state=1)
for i in range(12):
    stack.fit(X_train,Y_train[:,i])
    b_pred=stack.predict(X_test)
    b_pred=pd.DataFrame(b_pred)
    z=mean_absolute_error(Y_test[:,i], b_pred)
    m5.append(z)
print(m5)




