from flask import Flask, request, render_template, jsonify
from keras.preprocessing.image import image
import matplotlib.pyplot as plt
import numpy as np
app = Flask(__name__)
model = None
graph = None
import tensorflow as tf
from keras.models import load_model
def get_model():
    classifier = load_model("FH1.h5")
    return classifier
def load_model1():
    global graph
    graph = tf.get_default_graph()
    siamese_model = get_model()
    return siamese_model
def load_image(img_path, show=True):
    img_original = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    if show:
        plt.imshow(img_original)
        plt.axis('off')
        plt.show()
    return img_tensor
@app.route('/predict1', methods=['POST','GET'])
def predict1():

    with graph.as_default():
        pred = siamese_model.predict(new_image)
    print(pred)
    if pred < 0.5:
        a = "fen"
        print("分")
    else:
        print("合")
        a = "he"
    return jsonify(str(a))


if __name__ == '__main__':
    new_image = load_image(r"data/c/1.jpg")
    siamese_model = load_model1()
    app.run(
        host='0.0.0.0',
        port=8091,
        threaded=True,
    )