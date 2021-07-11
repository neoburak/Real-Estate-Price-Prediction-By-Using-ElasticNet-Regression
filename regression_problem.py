import pandas as pd
import  numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  accuracy_score, mean_squared_error,r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,LogisticRegressionCV,LassoCV,RidgeCV,ElasticNetCV,Ridge,Lasso,ElasticNet

import warnings
warnings.filterwarnings("ignore",category= DeprecationWarning)
warnings.filterwarnings("ignore",category= FutureWarning)



#problem determine house prices via 4 variable
df= pd.read_csv("Real estate.csv")

#print(df.isna().sum())
#print(df.isnull().sum())
#there are no null or na data
Q1= df["distance_to_the_nearest_MRT_station"].quantile(0.25)
Q3= df["distance_to_the_nearest_MRT_station"].quantile(0.75)
IQR= Q3-Q1

alt_sinir=  Q1 + 1.5*IQR
ust_sinir= Q3 + 1.5*IQR

aykiri_df= df["distance_to_the_nearest_MRT_station"]>ust_sinir
df_final= df["distance_to_the_nearest_MRT_station"][aykiri_df]=ust_sinir
#rint(df["distance_to_the_nearest_MRT_station"][aykiri_df])

"""
Index(['No', 'transaction_date', 'house_age',
       'distance_to_the_nearest_MRT_station', 'number_of_convenience_stores',
       'latitude', 'longitude', 'house price of unit area'],
"""
df_final= df.drop(columns=["No","transaction_date"])

#print(df)

#print(df.corr(method="pearson").sort_values(by=['house_price_of_unit_area'], ascending=False))

#sbn.regplot(data=df,x=df["house_age"],y=df["house_price_of_unit_area"])

#plt.show()

y= df_final["house_price_of_unit_area"]
X_= df_final.drop(columns=["house_price_of_unit_area","longitude","latitude"])

X_train,X_test,y_train,y_test= train_test_split(X_,y,random_state=10,test_size=0.33)

#lm= sm.OLS(y,X_).fit()
#print(lm.summary())
#Adj. R-squared (uncentered):              0.951 Prob(Omnibus):                  0.000

"""
def linModel(data,y,alg):
    model= alg().fit(X_train,y_train)
    y_pred= model.predict(X_test)
    RMSE= np.sqrt(mean_squared_error(y_test,y_pred))
    r2= r2_score(y_test,y_pred)
    model_name=alg.__name__
    return print(model_name + " Modeli Hatası: " + " "+ str(RMSE) + " R2_SCORE: "+str(r2))

models= [LinearRegression,Lasso,Ridge,ElasticNet]

for i in models:
    linModel(df,df["house_price_of_unit_area"],i)


"""




model= ElasticNet().fit(X_train,y_train)
y_pred= model.predict(X_test)
RMSE= np.sqrt(mean_squared_error(y_test,y_pred))
r2= r2_score(y_test,y_pred)
print("RMSE: "+ str(RMSE))
print("R2: "+ str(r2))


"""
elasticCV= ElasticNetCV(cv=15,n_jobs=-1,
                        random_state=10,l1_ratio=np.arange(0.1,1.0,0.1),
                        alphas=np.arange(0.2,5,0.2),n_alphas=np.array([5,10,15,50])).fit(X_train,y_train)
print(elasticCV.alpha_)
print(elasticCV.l1_ratio_)
#print(elasticCV.coef_)
#print(elasticCV.intercept_)
"""

model_final= ElasticNet(alpha=0.4,l1_ratio=0.3).fit(X_train,y_train)

y_pred_final= model_final.predict(X_test)

RMSE_final= np.sqrt(mean_squared_error(y_test,y_pred_final))
r2_final=r2_score(y_test,y_pred_final)
print("RMSE FİNAL:"+ str(RMSE_final) + "R2 Final: " + str(r2_final) )
#pd.Series(abs(svc_final.coef_[0]), index=X_train.columns).sort_values(ascending=True).plot(kind='barh')




#y= 46.48088700808442 + -0.27268012*X1 + -0.27268012*X2 + 0.89322467*X3

#ElasticNet Modeli Hatası:  8.855202855153188 R2_SCORE: 0.5290097217150582
#house_age
#distance_to_the_nearest_MRT_station
#number_of_convenience_stores
#print(model_final.coef_)

#df["distance_to_the_nearest_MRT_station"]


print(df_final.describe().T)
variable_graph= pd.Series(abs(model_final.coef_[0]), index=X_train.columns).sort_values(ascending=True).plot(kind='barh')
plt.subplot(2,2,1)
variable_box_house_age= sbn.boxplot(df_final["house_age"])
plt.subplot(2,2,2)
variable_box_distance_to_the_nearest_MRT_station= sbn.boxplot(df_final["distance_to_the_nearest_MRT_station"])
plt.subplot(2,2,3)
variable_box_number_of_convenience_stores= sbn.boxplot(df_final["number_of_convenience_stores"])
plt.subplot(2,2,4)
variable_box_number_of_convenience_stores= sbn.boxplot(df_final["number_of_convenience_stores"])

plt.suptitle("Modelin 4 değişkeni")

plt.show()

print("coefs: " + str(model_final.coef_))
print(model_final.coef_)
print("intercept: "+ str(model_final.intercept_))

print("Model Equation\n"+ "house_price_of_unit_area= 46.48088700808442+" + "-0.27268012* house_age+" + "-0.00778303* metro distance"
      + "0.89322467* number of stones" )

#print(model_final.predict([[12,1360,121]]))
#https://www.kaggle.com/quantbruce/real-estate-price-prediction