from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import json

TIME_FORMAT = '%Y-%m-%d %H:%M %Z'


def extract_weather_data(sample_data: dict) -> list:
    records = []
    for timestamp, data in sample_data.items():
        records.append(DailySummary(
            date=timestamp,
            air_temperature=data['air_temperature'],
            dew_point_t=data['dew_point_temperature'],
            pressure=data['pressure'],
            humidity=data['humidity']
        ))
    return records


def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


def read_sample():
    with open('sample_data.json', 'r') as data:
        file_data = json.load(data)
    return file_data


if __name__ == "__main__":
    json_data = read_sample()
    features = ["date", "air_temperature", "dew_point_t", "pressure", "humidity"]
    DailySummary = namedtuple("DailySummary", features)
    records = extract_weather_data(json_data)
    df = pd.DataFrame(records, columns=features).set_index('date')
    for feature in features:
        if feature != 'date':
            for N in range(1, 4):
                derive_nth_day_feature(df, feature, N)
    # print(df)
    # print(df.corr()[['air_temperature']].sort_values('air_temperature'))
    predictors = ['dew_point_t_2', 'humidity_1', 'humidity_3', 'pressure_1']
    df2 = df[['air_temperature'] + predictors]


    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [16, 22]

    # call subplots specifying the grid structure we desire and that
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True)

    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(4, 1)

    # use enumerate to loop over the arr 2D array of rows and columns
    # and create scatter plots of each meantempm vs each feature
    for row, col_arr in enumerate(arr):
        for col, feature in enumerate(col_arr):
            axes[row, col].scatter(df2[feature], df2['air_temperature'])
            if col == 0:
                axes[row, col].set(xlabel=feature, ylabel='air_temperature')
            else:
                axes[row, col].set(xlabel=feature)
    plt.show()


    # separate our my predictor variables (X) from my outcome variable y
    X = df2[predictors]
    y = df2['air_temperature']

    # Add a constant to the predictor variable set to represent the Bo intercept
    X = sm.add_constant(X)

    # (1) select a significance value
    alpha = 0.05

    # (2) Fit the model
    model = sm.OLS(y, X, missing='drop').fit()

    # (3) evaluate the coefficients' p-values
    model.summary()

    X = X.drop('humidity_1', axis=1)
    X = X.drop('humidity_3', axis=1)

    model = sm.OLS(y, X, missing='drop').fit()
    model.summary()

    X = X.drop('const', axis=1)

    X.fillna(X.mean(), inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # instantiate the regressor class
    regressor = LinearRegression()

    # fit the build the model by fitting the regressor to the training data
    regressor.fit(X_train, y_train)

    # make a prediction set using the test set
    prediction = regressor.predict(X_test)

    # Evaluate the prediction accuracy of the model
    from sklearn.metrics import mean_absolute_error, median_absolute_error

    print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
    print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
    print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

    joblib.dump(prediction, 'model_dump.sav')
