# imports
from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data,clean_data
from sklearn.model_selection import train_test_split
import joblib
from google.cloud import storage

### GCP configuration - - - - - - - - - - - - - - - - - - -
from TaxiFareModel.params import BUCKET_NAME,BUCKET_TRAIN_DATA_PATH,MODEL_NAME,MODEL_VERSION

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''defines the pipeline as a class attribute'''   
        # create dist pipeline
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('scaler', StandardScaler())
        ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_features', TimeFeaturesEncoder('pickup_datetime')),
            ('cat_transformer', OneHotEncoder())
        ])

        # create preprocessing pipeline
        time_features=['pickup_datetime']
        dist_features=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
    
        preprocessor =ColumnTransformer([
            ('dist_pipeline',dist_pipe,dist_features),
            ('time_pipeline', time_pipe,time_features)],
            remainder='drop')
    
        # Add pipeline containing the preprocessing and the regression model
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('linear_regression', LassoCV())])
        self.pipeline=pipeline
    
    
    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        model=self.pipeline
        return model.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        model=self.run()
        y_pred=model.predict(X_test)
        return compute_rmse(y_pred, y_test)
    
    
    def save_model(self,reg):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""
        
        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        joblib.dump(reg, 'model.joblib')
        # Implement here
        print("saved model.joblib locally")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        STORAGE_LOCATION=f'{MODEL_NAME}_{MODEL_VERSION}'
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')
        # Implement here
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df_cleaned=clean_data(df)
    # set X and y
    X=df_cleaned.drop(columns='fare_amount')
    y=df_cleaned.fare_amount
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # # train
    trainer=Trainer(X_train,y_train)
    trainer.run()
    # # evaluate
    print(trainer.evaluate(X_test, y_test))
    print('TODO')
