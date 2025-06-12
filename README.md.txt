# Plant Disease Detection System
A deep learning-powered web application built with Streamlit that can detect whether a plant leaf is healthy, affected by Early Blight, or Late Blight — just by uploading an image.

##  Features
* Detects plant diseases from leaf images using a trained neural network
* Clean and intuitive UI built with Streamlit
* Real-time image prediction and confidence visualization
* Lightweight and mobile-adaptable interface
* Built-in instructions and model info

##  Demo
![App Screenshot](screenshots/app_demo.png)

##  Model Details
* Input size: 128 x 128 RGB images
* Framework: TensorFlow / Keras
* Model File: `plant_disease_model.h5`
* Classes:
  * Early Blight
  * Late Blight
  * Healthy


##  Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit tensorflow pillow numpy
```


##  Running the App

Use the command below to launch the app locally:

```bash
streamlit run app.py
```
Then, open the local URL shown in your terminal (usually [http://localhost:8501](http://localhost:8501)).


##  How to Use
1. Upload a JPG/PNG image of a plant leaf from your system.
2. The app will display the image and process it.
3. It will show:
   * The predicted disease class (Early Blight, Late Blight, or Healthy)
   * The confidence score in percentage
   * A bar chart of confidence levels for all classes

##  Responsive Design
The layout is mobile-adaptable for smaller screens and has center-aligned titles, better spacing, and instructions in the sidebar.

## Project Structure

```
├── app.py                  # Main Streamlit application
├── plant_disease_model.keras  # Trained deep learning model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── screenshots/            # (Optional) Folder for app screenshots
```

##  Author
Made using Streamlit by Devesh Jha
Feel free to connect: [LinkedIn](http://www.linkedin.com/in/jha-devesh)


##  Notes
* This is a demo-level app intended for educational/portfolio purposes.
* The accuracy depends on the quality of the uploaded leaf image.
* For production use, further model optimization and testing are recommended.
