import joblib

def load_model(path):

    model = joblib.load(path)

    return model


def predict_price(model, input_data):

    prediction = model.predict(input_data)

    return prediction