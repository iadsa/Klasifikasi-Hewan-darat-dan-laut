from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__, template_folder='website', static_folder='hasil')

dic = {0: 'Anjing', 1: 'Ayam', 2: 'Bebek', 3: 'Bintang Laut',
       4: 'Hiu', 5: 'Ikan', 6: 'Kucing', 7: 'Kuda Laut',
       8: 'Macan', 9: 'Paus'}

model = load_model('Klasifikasi Hewan CNN.h5')


def predict_label(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = img.reshape((1, 224, 224, 3))
    prediction = model.predict(img)
    return dic[prediction.argmax()]


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        image_file = request.files['my_image']
        image_path = os.path.join('hasil', image_file.filename)
        image_file.save(image_path)
        prediction = predict_label(image_path)
        return render_template("index.html", prediction=prediction, img_path=image_file.filename)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
