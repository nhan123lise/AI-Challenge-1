from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import time
import tensorflow as tf
app = Flask(__name__)
dic = {0: "car", 1: "bird", 2: "bucket", 3: "clock"}

model = tf.keras.models.load_model('objects.h5')
model.make_predict_function()

@app.route('/')
def index():
    return render_template('drawing.html')

@app.route('/recognize', methods = ['POST'])
def recognize():

    if request.method == 'POST':
        data = request.get_json()
        imgBase64 = data['image']
        imgBytes = base64.b64decode(imgBase64)
        with open('temp.jpg','wb') as temp:
            temp.write(imgBytes)

        image = cv2.imread('temp.jpg')
        image = cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
        img_gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # img_prediction = np.expand_dims(img_gray,axis=-1)
        img_prediction = np.reshape(img_gray,(28,28,1))
        # img_prediction = (255 - img_gray.reshape(1, 28, 28).astype('float32'))/255

        print(img_prediction.shape)
        print(model.predict(np.array([img_prediction])))
        prediction = np.argmax(model.predict(np.array([img_prediction])),axis=-1)
        return jsonify({
            'prediction' : str(dic[prediction[0]]),
            'status' :True
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)