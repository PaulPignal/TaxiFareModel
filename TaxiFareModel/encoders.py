from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from TaxiFareModel.utils import haversine_vectorized

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        X_time=pd.DataFrame()
        X_time['time']=pd.to_datetime(X[self.time_column])
        X_time.index = X_time['time']
        X_time.index = X_time.index.tz_convert(self.time_zone_name)
        X_time["dow"] = X_time.index.weekday
        X_time["hour"] = X_time.index.hour
        X_time["month"] = X_time.index.month
        X_time["year"] = X_time.index.year
        return X_time[['dow', 'hour', 'month', 'year']].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self, 
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude", 
                 end_lat="dropoff_latitude", 
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        X_dist=X.copy()
        X_dist[['distance']]=haversine_vectorized(X_dist, 
                                    start_lat=self.start_lat,
                                    start_lon=self.start_lon,
                                    end_lat=self.end_lat,
                                     end_lon=self.end_lon)
        return X_dist[['distance']].reset_index(drop=True)
