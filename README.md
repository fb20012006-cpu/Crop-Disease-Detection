# Crop Disease Scanner

A mobile app to detect crop leaf diseases using AI.

## Features
- Detect 3 diseases: Healthy, Powdery, Rust
- Severity calculation
- LIME and SHAP AI explanations
- Camera and Gallery support

## Project Structure
```
CropDisease/
├── backend/
│   ├── app.py                  # Flask server
│   ├── requirements.txt        # Python packages
│   └── model/
│       └── crop_disease_model.h5  # Trained model
└── leaf_scanner/
    └── lib/
        ├── main.dart                       # Flutter UI
        └── services/
            └── disease_api_service.dart    # API service

```

## How to Run

### Backend (Flask Server)
1. Open terminal in backend folder
2. Activate virtual environment:
   ```
   venv\Scripts\activate
   ```
3. Run server:
   ```
   python app.py
   ```
4. Server runs at `http://localhost:5000`

### Flutter App
1. Connect phone via USB
2. Enable USB Debugging on phone
3. Open terminal in leaf_scanner folder
4. Run:
   ```
   flutter run -d YOUR_DEVICE_ID
   ```

## Model Details
- Architecture: MobileNetV2
- Input size: 224x224
- Classes: Healthy, Powdery, Rust
- Framework: TensorFlow/Keras

## Requirements
- Python 3.x
- Flutter 3.x
- Android phone with USB debugging enabled
- Laptop and phone on same WiFi network

## Notes
- Change IP address in `disease_api_service.dart` if your WiFi IP changes
- Run `ipconfig` to get your current IP address
- Model file must be in `backend/model/crop_disease_model.h5`
# Crop-Disease-Detection
This project is an AI-powered crop disease detection system that identifies plant diseases from leaf images using deep learning. The system helps farmers and researchers quickly detect plant diseases and understand the prediction using explainable AI techniques.
