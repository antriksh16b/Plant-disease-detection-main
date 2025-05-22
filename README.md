cat <<EOF > README.md
# 🌿 Plant Disease Detection

A deep learning-based web application to detect plant diseases from leaf images using a trained Convolutional Neural Network (CNN) model.

---

## 🚀 Features

- 📷 Upload plant leaf images
- 🧠 Predict plant diseases using a trained CNN model
- 🌐 User-friendly web interface
- 📊 Displays prediction confidence

---

## 🛠️ Tech Stack

| Frontend | Backend | ML/AI              | Other         |
|----------|---------|--------------------|---------------|
| HTML/CSS | Flask   | TensorFlow, Keras  | OpenCV, NumPy |

---

## 📦 Installation

1. Clone the repository:

   \`\`\`
   git clone https://github.com/antriksh16b/Plant-disease-detection-main.git
   cd Plant-disease-detection-main
   \`\`\`

2. Install dependencies:

   \`\`\`
   pip install -r requirements.txt
   \`\`\`

3. Run the app:

   \`\`\`
   python app.py
   \`\`\`

4. Open your browser and visit:

   \`\`\`
   http://127.0.0.1:5000/
   \`\`\`

---

## 🧪 Dataset

This project uses the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), which contains over 50,000 images of healthy and diseased plant leaves, collected under controlled conditions.

---

## 📷 Screenshots

> *(Replace these with your actual screenshots)*

### Upload Page

![Upload Page](screenshots/upload_page.png)

### Prediction Result

![Prediction Result](screenshots/prediction_result.png)

---

## 📁 Project Structure

\`\`\`
Plant-disease-detection-main/
│
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates (Jinja2)
├── model/                # Trained ML model (.h5 file)
├── app.py                # Flask application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
\`\`\`

---

## ⭐ Show Your Support

If you like this project:

- Give it a ⭐ on GitHub
- Fork it to your own profile
- Share feedback or contribute!

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).
EOF
