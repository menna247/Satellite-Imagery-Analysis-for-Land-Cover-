from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import pandas as pd
import copy

app = Flask(__name__)
model = load_model('DeepGlobe/result/model/unet_model_v3_epoch6.h5', compile=False)
imsize = model.input_shape[1]

class_df = pd.DataFrame({
    'name': ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown'],
    'r': [0, 255, 255, 0, 0, 255, 0],
    'g': [255, 255, 0, 255, 0, 255, 0],
    'b': [255, 0, 255, 0, 255, 255, 0]
})

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('jpg', 'png'))

def rescale(img):
    if np.max(img) > 1:
        img = np.multiply(img, 1.0 / 255)
    return img

def one_hot_encode(img, class_map=None):
    if class_map is None:
        class_map = class_df

    img_copy = copy.deepcopy(img)
    frame = np.zeros((img.shape[0], img.shape[1], len(class_map))).astype('int')

    for class_channel, (index, row) in enumerate(class_map.iterrows()):
        new_img = copy.deepcopy(img_copy)
        R = new_img[:, :, 0]
        G = new_img[:, :, 1]
        B = new_img[:, :, 2]

        mask = (R == row['r']) & (G == row['g']) & (B == row['b'])
        frame[:, :, class_channel] = mask.astype(int)

    return frame

def preprocessor_images(image, b_threshold=128):
    final_img = rescale(image)
    return final_img

def preprocessor_masks(image, b_threshold=128, class_map=None):
    image = one_hot_encode(image, class_map)
    final_img = rescale(image)
    return final_img

def predict_mask(image):
    image_array = preprocessor_images(image)
    image_array = np.expand_dims(image_array, axis=0)
    predicted_mask = model.predict(image_array)
    predicted_mask = np.argmax(predicted_mask, axis=-1)

    mask_color = np.zeros((predicted_mask.shape[1], predicted_mask.shape[2], 3), dtype=np.uint8)

    most_frequent_class = np.bincount(predicted_mask.flatten()).argmax()
    predicted_land_type = class_df.iloc[most_frequent_class]['name']

    for class_index in range(predicted_mask.max() + 1):
        mask_color[predicted_mask[0] == class_index] = [
            class_df.iloc[class_index]['r'],
            class_df.iloc[class_index]['g'],
            class_df.iloc[class_index]['b']
        ]

    return mask_color, predicted_land_type

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            image = Image.open(file.stream).convert('RGB')
            mask, predicted_land_type = predict_mask(image)
            mask_image = Image.fromarray(mask)
            mask_image_path = os.path.join('static', 'predicted_mask.png')
            mask_image.save(mask_image_path, 'PNG')
            return render_template('result.html', mask_image_path=mask_image_path, predicted_land_type=predicted_land_type)
        else:
            return "Please upload a valid JPG or PNG file", 400
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)