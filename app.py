from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

categorical_cols = ['cut', 'color', 'clarity']
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

columns = categorical_cols + numerical_cols
cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
drop_down_order = [cut_categories, color_categories, clarity_categories]


@app.route("/")
def index():
    return render_template("index.html", col=columns, drop_down_order=drop_down_order, categorical_cols=categorical_cols, enumerate=enumerate)


@app.route("/predict", methods=["GET", "POST"])
def new_prediction():
    cut = request.form["cut"].title()
    color = request.form["color"].upper()
    clarity = request.form["clarity"].upper()
    carat = float(request.form["carat"])
    depth = float(request.form["depth"])
    table = float(request.form["table"])
    x = float(request.form["x"])
    y = float(request.form["y"])
    z = float(request.form["z"])
    data = CustomData(carat=carat, color=color, depth=depth, table=table, clarity=clarity, cut=cut, x=x, y=y, z=z)
    df = data.get_data_as_dataframe()
    model = PredictPipeline()
    prediction = model.predict(df)
    return render_template("result.html", predict=str(round(prediction[0], 2)))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
