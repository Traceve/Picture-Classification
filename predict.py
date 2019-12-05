import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
#classifier=load_model("FH1.h5")


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

new_image = load_image(r"data/c/1.jpg")
pred= siamese_model.predict(new_image)
print(pred)

if pred<0.5 : print("分")
else : print("合")
