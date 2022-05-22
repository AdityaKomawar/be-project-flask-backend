import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications import vgg16
from keras.utils.data_utils import get_file
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin
import cv2


app = Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def get_model():
  global model
  file_path = get_file("h5_model", "https://beproject.blob.core.windows.net/model/vgg16_model_with_96_acc_35_epoch.h5")
  model = load_model(file_path)
  print("====Model Loaded!")

def preprocess_image(image, target_size, res):
  res['mode'] = image.mode
  res['shape'] = img_to_array(image).shape
  if image.mode != "RGB":
    image = image.convert("RGB")
  image = image.resize(target_size, Image.LANCZOS)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)

  return image

# def inverse_classes(num):
#   if num == 0:
#     return 'Glioma Tumor'
#   elif num == 1:
#     return 'Meningioma Tumor'
#   elif num == 2:
#     return 'No Tumor'
#   else:
#     return 'Pituitary Tumor'

print("====Loading keras model...")
get_model()

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
  res = {}
  processed_image = ""
  try:
    message = request.get_json()
    # print(message['image'].strip())
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    # print(decoded)
    image = Image.open(io.BytesIO(decoded))
    # print("====Inside route!")
    processed_image = preprocess_image(image, target_size=(224, 224), res=res)
    # processed_image = cv2.resize(preprocess_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    processed_image = vgg16.preprocess_input(processed_image.copy())
    res['prediction'] = {}
  except Exception as e:
    # print(traceback.format_exc())
    res['error'] = 'error: ' + str(e),
  else:
    prediction = model.predict(np.reshape(processed_image, (-1, 224, 224, 3))).tolist()

    res['prediction']['glioma'] = round(prediction[0][0], 4)
    res['prediction']['meningioma'] = round(prediction[0][1], 4)
    res['prediction']['no_tumor'] = round(prediction[0][2], 4)
    res['prediction']['pituitary'] = round(prediction[0][3], 4)

  return jsonify(res)



