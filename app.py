import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import random
import time
import sys
import cv2
import numpy as np
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Random import get_random_bytes
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA256
import hashlib

salt = b'MyFixedSalt123'
keySize = 16
ivSize = AES.block_size

app = Flask(__name__)

model = load_model('brain_tumor_classification_VGG16.h5')
model1 = load_model('brain21.h5')

def get_className(classNo1):
    if classNo1 == 1:
        return "Brain tumor"
    elif classNo1 == 2:
        return "Not a brain tumor"
    elif classNo1 == 3:
        return "Not a brain image"

def add_red_patch(image_path):
    img = cv2.imread(image_path)
    patch = np.array([[[0, 0, 255]] * 3] * 3, dtype=np.uint8)  # Red patch

    # Random coordinates within -20 to -10 range from the bottom-left corner
    y_coord = random.randint(-20, -10)
    x_coord = random.randint(-20, -10)

    # Ensure coordinates are within the image boundaries
    y_start = max(0, img.shape[0] + y_coord)
    x_start = max(0, img.shape[1] + x_coord)

    img[y_start:y_start+3, x_start:x_start+3] = patch
    cv2.imwrite(image_path, img)

def encrypt(file_path, keyInput):
    start_time = time.time()
    imageOrig = cv2.imread(file_path)
    rowOrig, columnOrig, depthOrig = imageOrig.shape
    minWidth = (AES.block_size + AES.block_size) // depthOrig + 1
    if columnOrig < minWidth:
        print('The minimum width of the image must be {} pixels, so that IV and padding can be stored in a single additional row!'.format(minWidth))
        sys.exit()
    imageOrigBytes = imageOrig.tobytes()
    key = PBKDF2(keyInput, salt, dkLen=16, count=1000, prf=lambda p,s: hashlib.pbkdf2_hmac('sha256', p, s, 1000))
    hash_obj = SHA256.new(key)
    iv = hash_obj.digest()[:ivSize]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    imageOrigBytesPadded = pad(imageOrigBytes, AES.block_size)
    ciphertext = cipher.encrypt(imageOrigBytesPadded)
    paddedSize = len(imageOrigBytesPadded) - len(imageOrigBytes)
    void = columnOrig * depthOrig - ivSize - paddedSize
    ivCiphertextVoid = iv + ciphertext + bytes(void)
    imageEncrypted = np.frombuffer(ivCiphertextVoid, dtype=imageOrig.dtype).reshape(rowOrig + 1, columnOrig, depthOrig)
    cv2.imwrite(file_path, imageEncrypted)
    cv2.destroyAllWindows()

def decrypt(file_path, keyInput):
    key = PBKDF2(keyInput, salt, dkLen=16, count=1000, prf=lambda p,s: hashlib.pbkdf2_hmac('sha256', p, s, 1000))
    hash_obj = SHA256.new(key)
    iv = hash_obj.digest()[:ivSize]
    imageEncrypted = cv2.imread(file_path)
    rowEncrypted, columnOrig, depthOrig = imageEncrypted.shape
    rowOrig = rowEncrypted - 1
    encryptedBytes = imageEncrypted.tobytes()
    iv = encryptedBytes[:ivSize]
    imageOrigBytesSize = rowOrig * columnOrig * depthOrig
    paddedSize = (imageOrigBytesSize // AES.block_size + 1) * AES.block_size - imageOrigBytesSize
    encrypted = encryptedBytes[ivSize: ivSize + imageOrigBytesSize + paddedSize]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decryptedImageBytesPadded = cipher.decrypt(encrypted)
    decryptedImageBytes = unpad(decryptedImageBytesPadded, AES.block_size)
    decryptedImage = np.frombuffer(decryptedImageBytes, imageEncrypted.dtype).reshape(rowOrig, columnOrig, depthOrig)
    cv2.imwrite(file_path,decryptedImage)
    cv2.destroyAllWindows()
    return decryptedImage

def getResult(img, keyInput):
    image = decrypt(img, keyInput)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    predict_x = model1.predict(input_img)
    result = 0
    if result == 0:
        predict_y = model.predict(input_img)
        print(predict_y)
        results = [[i, r] for i, r in enumerate(predict_y)]
        results.sort(key=lambda x: x[1], reverse=True)
        for r in results:
            print(str(r[1][1]*100))
        output = np.argmax(predict_y, axis=1)
        if output == 1:
            return 1
        else:
            return 2
    else:
        return 3

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        basepath = os.path.dirname(__file__)
        filename = request.form.get('textInput')
        keyInput = str(filename)
        filename = str(filename) + '.png'
        print(filename)
        file_path = os.path.join(basepath, 'uploads', filename)
        value = getResult(file_path, keyInput)
        res = get_className(value)
        return res
    else:
        return render_template('predict.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        fn = request.form.get('textInput')
        keyInput = str(fn)
        filename = str(fn) + '.png'
        protection_status = keyInput[8]!='0'
        print(filename)
        file_path = os.path.join(basepath, 'uploads', filename)
        f.save(file_path)
        print(protection_status)
        
        if not protection_status:
            add_red_patch(file_path)
        
        encrypt(file_path, keyInput)
        res = fn
        return res
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=False)
