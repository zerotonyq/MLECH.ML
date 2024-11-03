from DataTransform import DataTransform
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_test', required=True, type=str, help='path to data for test model before deployment')
    parser.add_argument('--raw_data', required=True, type=str, help='path to raw data')
    parser.add_argument('--not_save_test', action='store_true', default=False, help='is save test data')
    parser.add_argument('--not_save_train', action='store_true', default=False, help='is save train data')
    parser.add_argument('--not_save_model_test', action='store_true', default=False, help='is save data for testing model')
    parser.add_argument('--out_train', type=str, required=False, default='data/features/prepared_train.csv', help='path for save train data')
    parser.add_argument('--out_test', type=str, required=False, default='data/features/prepared_test.csv', help='path for save test data')
    parser.add_argument('--out_model_test', type=str, required=False, default='data/processed/test.csv', help='path for save model test data')

    return parser.parse_args()


def load_data(path):
    return pd.read_csv(path, parse_dates=True)


def data_filter(df, lambda_func):
    return df[df.apply(lambda_func, axis=1)]



def main():
    args = parse_args()
    car_train = pd.read_csv(args.raw_data + 'car_train.csv')
    model_test = pd.read_csv(args.model_test)
    car_test = pd.read_csv(args.raw_data + 'car_test.csv')
    fix_info = pd.read_csv(args.raw_data + 'fix_info.csv')
    # driver_info = pd.read_csv(args.raw_data + 'driver_info.csv')
    rides_info = pd.read_csv(args.raw_data + 'rides_info.csv')

    ri_filter_func = lambda row: row['ride_duration'] < 60 * 5 and row['ride_cost'] < 4000 and row['distance'] < 7000
    rides_info = data_filter(rides_info, ri_filter_func)

    GROUPBY_COLUMN = ['car_id']

    # fix_info transformarion
    fix_info_cols = ['car_id', 'destroy_degree', 'work_duration']
    fix_info_aggregation = {
        'destroy_degree': ['mean', 'median', 'count'],
        'work_duration': ['sum', 'mean']
    }
    fix_info_transformer = DataTransform(fix_info,
                                         group_cols=GROUPBY_COLUMN,
                                         aggregate_functions=fix_info_aggregation)

    # rides_info transformation
    rides_info_cols = ['car_id', 'rating', 'ride_duration', 'ride_cost', 'speed_avg', 'speed_max', 'stop_times',
                       'distance',
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

    if not args.not_save_test:
        features = car_train.merge(transformed_fix_info, how='left', on='car_id')
        features = features.merge(transformed_rides_info, how='left', on='car_id')
        features.to_csv(args.out_train, index=False, encoding='utf-8', header=True)

    if not args.not_save_train:
        features_test = car_test.merge(transformed_fix_info, how='left', on='car_id')
        features_test = features_test.merge(transformed_rides_info, how='left', on='car_id')
        features_test.to_csv(args.out_test, index=False, encoding='utf-8', header=True)

    if not args.not_save_model_test:
        features_model_test = model_test.merge(transformed_fix_info, how='left', on='car_id')
        features_model_test = features_model_test.merge(transformed_rides_info, how='left', on='car_id')
        features_model_test.to_csv(args.out_model_test, index=False, encoding='utf-8', header=True)


if __name__ == '__main__':
    main()



