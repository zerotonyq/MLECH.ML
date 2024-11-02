from DataTransform import DataTransform
import pandas as pd

SAVE_PATH = '../../data/features/prepared_train.csv'
DATA_PATH = '../../data/raw/'
GROUPBY_COLUMN = ['car_id']

def filter_rides_info(rides_info: pd.DataFrame) -> pd.DataFrame:
    tmp = rides_info.copy(deep=True)
    tmp = tmp[tmp['ride_duration'] < 60 * 5]
    tmp = tmp[tmp['distance'] < 7000]
    tmp = tmp[tmp['ride_cost'] < 4000]
    return tmp


rides_info = pd.read_csv(DATA_PATH + 'rides_info.csv')
car_train = pd.read_csv(DATA_PATH + 'car_train.csv')
car_test = pd.read_csv(DATA_PATH + 'car_test.csv')
fix_info = pd.read_csv(DATA_PATH + 'fix_info.csv')
driver_info = pd.read_csv(DATA_PATH + 'driver_info.csv')

rides_info = filter_rides_info(rides_info)

fix_info_cols = ['car_id', 'destroy_degree', 'work_duration']
fix_info_aggregation = {
    'destroy_degree': ['mean', 'median', 'count'],
    'work_duration': ['sum', 'mean']
}
fix_info_transformer = DataTransform(fix_info,
                                     group_cols=GROUPBY_COLUMN,
                                     aggregate_functions=fix_info_aggregation)

rides_info_cols = ['car_id', 'rating', 'ride_duration', 'ride_cost', 'speed_avg', 'speed_max', 'stop_times', 'distance',
                   'refueling', 'user_ride_quality', 'deviation_normal']
rides_info_aggregation = {
    'rating': ['mean', 'median'],
    'ride_duration': ['mean', 'median', 'sum'],
    'ride_cost': ['mean', 'median'],
    'speed_avg': ['mean', 'median'],
    'speed_max': ['mean', 'median'],
    'stop_times': ['mean', 'median', 'sum'],
    'distance': ['mean', 'median', 'sum'],
    'refueling': ['sum'],
}
rides_info_transformer = DataTransform(rides_info,
                                       group_cols=GROUPBY_COLUMN,
                                       aggregate_functions=rides_info_aggregation)


transformed_fix_info = fix_info_transformer.transform(fix_info_cols)
transformed_rides_info = rides_info_transformer.transform(rides_info_cols)

features = car_train.merge(transformed_fix_info, how='left', on='car_id')
features = features.merge(transformed_rides_info, how='left', on='car_id')

features.to_csv(SAVE_PATH, index=False, encoding='utf-8', header=True)