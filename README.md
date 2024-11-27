# Deep Fake Detection

A **Streamlit-based web application** leveraging deep learning to identify DeepFake videos. This app uses a combination of **ResNeXt** and **LSTM** to analyze video frames and predict whether a video is real or fake, along with confidence scores.

## Features
- Upload video files to detect DeepFake content.
- Real-time face detection and frame processing.
- Generates predictions with visual heatmaps for insights.
- Simple, intuitive interface powered by Streamlit.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sanya003/Deep-Fake-Detection.git
   cd Deep-Fake-Detection

2. Install dependencies:
   - Create a virtual environment (optional):
       ```bash
       python -m venv venv
       venv\Scripts\activate
   - Install required packages:
       ```bash
       pip install -r requirements.txt

3. Download the model:
   - Place the model file (model.pt) in the root directory of the project.
   - Update the path in app.py if needed.


## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py

2. Open the app in your browser at http://localhost:8501.

3. Upload a video file and click "Process" to analyze its authenticity. View predictions and confidence scores directly in the app.


## File Structure

```plaintext
Deep-Fake-Detection/
│
├── app.py                
├── function.py            
├── model.pt               
├── requirements.txt       
└── README.md              
