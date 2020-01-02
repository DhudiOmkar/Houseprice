#change working folder
os.chdir('C:\\Users\\domkar\\Desktop\\Deploy\\Deploy')
#Data inporting
Data=pd.read_csv('kc_house_data.csv')
Data.head()
#Removing variables
Data.drop(Data.iloc[:,[0,1]],axis=1, inplace=True)
#Saparating X and Y variables
y=Data.iloc[:,0]
x=Data.drop(Data.iloc[:,[0]],axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=1)
#Fitting model
from sklearn.ensemble import GradientBoostingRegressor
ml=GradientBoostingRegressor().fit(x_train, y_train)
y_pred=ml.predict(x_test)
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(y_test, y_pred))
RMSE

# Saving model to disk
pickle.dump(ml, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))