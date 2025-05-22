import os
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import argmax
from PIL import Image
from keras.preprocessing import image
import streamlit as st
from pathlib import Path
from utils import label_map_util
from utils import visualization_utils as vis_util

# TensorFlow setup
tf.executing_eagerly()  # TensorFlow 2.x eager execution enabled by default

MODEL_NAME = './object_detection/inference_graph'
IMAGE_NAME = './object_detection/images/out.jpg'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = './object_detection/training/labelmap.pbtxt'
NUM_CLASSES = 6

# Load labels and model
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Input and output tensors for detection
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Dictionary for plant disease treatments
TREATMENTS = {
    'Apple___Apple_scab': 'Apply fungicide sprays like captan or mancozeb.',
    'Apple___Black_rot': 'Prune infected branches and apply fungicides.',
    'Apple___Cedar_apple_rust': 'Remove nearby cedar trees or use resistant varieties.',
    'Apple___healthy': 'No treatment needed.',
    'Blueberry___healthy': 'No treatment needed.',
    'Cherry___healthy': 'No treatment needed.',
    'Cherry___Powdery_mildew': 'Use sulfur-based fungicides and proper pruning.',
    'Grape___Black_rot': 'Use protective fungicide sprays and remove mummies.',
    'Grape___Esca_Black_Measles': 'Avoid pruning during wet conditions and use clean tools.',
    'Grape___healthy': 'No treatment needed.',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot': 'Apply protective fungicides.',
    'Orange___Haunglongbing': 'No cure, remove infected trees and control psyllids.',
    'Peach___Bacterial_spot': 'Use copper-based sprays during dormant season.',
    'Peach___healthy': 'No treatment needed.',
    'Pepper_bell___Bacterial_spot': 'Use certified disease-free seeds and apply copper sprays.',
    'Pepper_bell___healthy': 'No treatment needed.',
    'Potato___Early_blight': 'Apply chlorothalonil-based fungicides.',
    'Potato___healthy': 'No treatment needed.',
    'Potato___Late_blight': 'Apply metalaxyl-based fungicides immediately.',
    'Raspberry___healthy': 'No treatment needed.',
    'Soybean___healthy': 'No treatment needed.',
    'Squash___Powdery_mildew': 'Use sulfur or neem oil sprays.',
    'Strawberry___healthy': 'No treatment needed.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves and apply fungicides.'
}

def load_and_process_image(image_path, target_size=(150, 150)):
    """Loads and preprocesses image for classification"""
    img = image.load_img(image_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return np.vstack([x])

def classify_plant_disease(image_path):
    """Classify the plant disease based on the image."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(24, activation='softmax')
    ])
    model.load_weights("./object_classification/rps.h5")  # Load pretrained model
    x = load_and_process_image(image_path)
    classes = model.predict(x, batch_size=10)
    result = argmax(classes)
    LABELS = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew', 'Grape___Black_rot',
        'Grape___Esca_Black_Measles', 'Grape___healthy', 'Grape___Leaf_blight_Isariopsis_Leaf_Spot',
        'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper_bell___Bacterial_spot',
        'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy',
        'Strawberry___Leaf_scorch'
    ]
    disease_name = LABELS[result]
    treatment = TREATMENTS.get(disease_name, "No treatment information available.")
    return disease_name, treatment

# Streamlit UI
def main():
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Plant Disease Detection & Classification")
    st.text("Build with Streamlit and TensorFlow")

    activities = ["About", "Plant Disease"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    enhance_type = st.sidebar.radio("Type", ["Detection", "Classification", "Treatment"])

    if choice == 'About':
        intro_markdown = Path("./doc/about.md").read_text()
        st.markdown(intro_markdown, unsafe_allow_html=True)

    if choice == 'Plant Disease':
        image_file = st.file_uploader("Upload Image", type=['jpg'])
        if image_file is not None:
            image_path = './object_classification/images/out.jpg'
            image = Image.open(image_file)
            image.save(image_path)

            if enhance_type == 'Detection':
                st.header("Plant Disease Detection")
                # Process image using object detection logic
                in_image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
                image_expanded = np.expand_dims(image_rgb, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded}
                )
                vis_util.visualize_boxes_and_labels_on_image_array(
                    in_image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores), category_index, use_normalized_coordinates=True,
                    line_thickness=8, min_score_thresh=0.60
                )
                st.image(in_image, channels='RGB')

            elif enhance_type == 'Classification':
                st.header("Plant Disease Classification")
                disease_name, treatment = classify_plant_disease(image_path)
                st.image(image_path, use_column_width=True)
                st.write(f"**Plant Disease**: {disease_name}")
                st.write(f"**Treatment**: {treatment}")
                st.success("Classification completed successfully!")

            elif enhance_type == 'Treatment':
                st.header("Plant Disease Treatment Suggestions")
                disease_name, treatment = classify_plant_disease(image_path)
                st.write(f"**Disease**: {disease_name}")
                st.write(f"**Suggested Treatment**: {treatment}")

if __name__ == "__main__":
    main()
