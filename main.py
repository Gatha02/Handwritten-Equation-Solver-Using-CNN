#C:\\Users\\Gatha\\
import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json

# Function to load the trained model
def load_model():
    json_file = open('C:\\Users\\Gatha\\model_final.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("C:\\Users\\Gatha\\model_final.h5")
    return loaded_model

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    train_data = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        im_crop = thresh[y:y+h+10, x:x+w+10]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (28, 28, 1))
        train_data.append(im_resize)
    return train_data

# Function to predict the equation
def predict_equation(model, train_data):
    equation = ''
    for i in range(len(train_data)):
        train_data[i] = np.array(train_data[i])
        train_data[i] = train_data[i].reshape(1, 28, 28, 1)
        result = np.argmax(model.predict(train_data[i]), axis=-1)
        if result[0] == 10:
            equation += '-'
        elif result[0] == 11:
            equation += '+'
        elif result[0] == 12:
            equation += '*'
        else:
            equation += str(result[0])
    return equation

def main():
    st.title("Handwritten Equation Solver")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.write("")
        st.write("Classifying...")

        # Load the model
        model = load_model()

        # Preprocess the image
        train_data = preprocess_image(image)

        # Predict the equation
        equation = predict_equation(model, train_data)

        st.write("Predicted Equation:", equation)

        # Evaluate the equation and display the result
        try:
            result = eval(equation)
            st.write("Result:", result)
        except Exception as e:
            st.write("Error occurred while evaluating the equation.")

if __name__ == '__main__':
    main()
