from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings('ignore')
import datacube
import io
import odc.algo
import matplotlib.pyplot as plt
from datacube.utils.cog import write_cog
import base64
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from deafrica_tools.plotting import display_map, rgb
dc = datacube.Datacube(app="04_Plotting")
# 15.85828652, 80.78694696
# 15.75418332, 81.02203692

app = Flask(__name__)

@app.route("/")
def hello_world():
        # Read the merged CSV file
    df = pd.read_csv('2017-2023.csv')

    # Select the relevant columns for training
    features = ['month', 'year']
    target = ['mangrove', 'regular', 'closed', 'healthy', 'unhealthy']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    model = RandomForestRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Prepare the data for plotting
    df_plot = pd.DataFrame({'Year': X_test['year'], 'Month': X_test['month'], 'Actual mangrove': y_test['mangrove'],
                            'Predicted mangrove': y_pred[:, 0]})

    # Sort the data by year and month
    df_plot = df_plot.sort_values(['Year', 'Month'])

    # Create a single index combining year and month
    df_plot['Year-Month'] = df_plot['Year'].astype(str) + '-' + df_plot['Month'].astype(str)

    # Generate the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['Year-Month'], df_plot['Actual mangrove'], 'o-', label='Actual mangrove_area')
    plt.plot(df_plot['Year-Month'], df_plot['Predicted mangrove'], 'o-', label='Predicted mangrove_area')
    plt.xlabel('Year-Month')
    plt.ylabel('Area')
    plt.title('Changes in Mangrove Areas Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/plot.png')  # Save the plot as a file
    return render_template("index.html")

@app.route('/my_flask_route', methods=['GET', 'POST'])
def my_flask_function():
    if request.method == "POST":
        lmin = request.json['lat_min']
        lmax = request.json['lat_max']
        lnmin = request.json['lng_min']
        lnmax = request.json['lng_max']
        ty = request.json['t']

        lat_range = (lmin, lmax)
        lon_range = (lnmin, lnmax)
        print(lat_range, lon_range)
        time_range = ('2019-01-15', '2023-05-15')
        # display_map(x=lon_range, y=lat_range)
        try:
            ds = dc.load(product="s2a_sen2cor_granule",
                            measurements=["B04_10m","B03_10m","B02_10m", "B08_10m", "SCL_20m", "B11_20m"],
                        x=lon_range,
                        y=lat_range,
                        time=time_range,
                        output_crs='EPSG:6933',
                        resolution=(-30, 30))
            dataset = ds
            dataset =  odc.algo.to_f32(dataset)
            band_diff = dataset.B08_10m - dataset.B04_10m
            band_sum = dataset.B08_10m + dataset.B04_10m
        except Exception as e:
            return jsonify({'error': "No Data Found"})

        # Calculate NDVI and store it as a measurement in the original dataset
        if (ty=='ndvi'):
            ndvi = band_diff / band_sum
            plt.figure(figsize=(8, 8))
            ndvi_subplot=ndvi.isel(time=[0,-1])
            ndvi_subplot.plot(col='time', cmap='YlGn', vmin=-1, vmax=1, col_wrap=2)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            # plt.savefig('./static/my_plot.png')
            # Serve the image file in the Flask app
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        elif (ty=='mvi'):
            mvi = (dataset.B08_10m - dataset.B03_10m) / (dataset.B11_20m - dataset.B03_10m)
            plt.figure(figsize=(8, 8))
            mvi_subplot=mvi.isel(time=[0,-1])
            mvi_subplot.plot(col='time', cmap='cividis', vmin=1, vmax=20, col_wrap=2)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            # plt.savefig('./static/my_plot.png')
            # Serve the image file in the Flask app
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        


    # Return the base64 encoded PNG image as JSON
        return jsonify({'image': img_base64})
    # Calculate the components that make up the NDVI calculation

if __name__ == '__main__':
    app.run(port = 5002,debug=False)
