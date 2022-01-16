# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def main():
    # pandas setup: elimination of the limit of column displayed
    pd.set_option('display.max_columns', None)

    # reading data
    data = pd.read_csv("../housing.csv")

    # data cleaning
    # checking if dataframe contains any empty records
    print(data.isna().sum())
    print(data["total_bedrooms"].median())
    # fills empty records with median of that column
    data["total_bedrooms"].fillna(data["total_bedrooms"].median(), inplace=True)
    print(data.isna().sum())

    # checking if dataframe contains any duplicated rows
    print(data.duplicated().any())

    # drawing California map with houses and their values marked
    map = Basemap(projection='lcc', resolution='i', lat_0=37.5, lon_0=-119, width=1E6, height=1.2E6)
    map.drawcoastlines(color='gray')
    map.drawcountries(color='gray')
    map.drawstates(color='gray')
    map.scatter(data["longitude"], data["latitude"], latlon=True, alpha=0.5, c=data["median_house_value"], cmap="jet")
    plt.colorbar(label="House Value")
    plt.show()

    # computing pairwise correlation of columns and drawing heatmap, excluding column containing non numerical values
    data_corr = data.corr()
    sb.heatmap(data_corr, annot=True, cmap="Purples")
    plt.show()

    # printing sorted house value correlation
    print(data_corr["median_house_value"].sort_values(ascending=False))

    # dropping not useful data
    data = data.drop(columns=["longitude", "latitude"])

    # scalling data
    numerical_data = data.drop("ocean_proximity", axis=1)
    scaler = StandardScaler()
    scaling = scaler.fit(numerical_data)
    scaled_data = scaling.transform(numerical_data)
    scaled_data = pd.DataFrame(data=scaled_data, columns=numerical_data.columns)
    print(scaled_data)

    # adding ocean_proximity data seperated into different columns
    ocean_proximity = data["ocean_proximity"]
    ocean_proximity = pd.get_dummies(ocean_proximity, drop_first=True)
    new_data = pd.concat([scaled_data, ocean_proximity], axis=1)
    print(data)

    # calculating multiple linear regression
    # assigning data to x and y
    x = new_data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    # spliting the data into two groups: for training and for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
    model = LinearRegression().fit(x_train, y_train)
    print("Intercept: ", model.intercept_)
    print("Coefficients: ", list(zip(x, model.coef_)))

    # predictions
    y_pred = model.predict(x_test)

    # plotting actual values compared to predicted ones
    plt.figure(figsize=(15,10))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted")
    plt.show()

    # actual values vs predicted in array form
    print(pd.DataFrame({"Actual Value": y_test, "Predicted Value": y_pred, "Difference": y_test - y_pred}))

    # printing r-squared
    print("R-squared: ", r2_score(y_test, y_pred))

    # calcalating and printing mean_absolute_error
    predict_house_value = model.predict(x)
    mae = mean_absolute_error(y, predict_house_value)
    print("Mean absolute error: ", mae)


if __name__ == "__main__":
    main()