import os
import io
import cv2
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, render_template_string
import torchvision.transforms as transforms

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. PHOTO MODEL (ImageCNN)
# ==========================================
class ImageCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

photo_model = ImageCNN(num_classes=2).to(device)
try:
    photo_model.load_state_dict(torch.load("image_cnn_weights.pth", map_location=device))
    photo_model.eval()
except FileNotFoundError:
    print("Warning: image_cnn_weights.pth not found.")

photo_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])

# ==========================================
# 2. VIDEO MODEL (CustomCNN)
# ==========================================
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

video_model = CustomCNN().to(device)
try:
    video_model.load_state_dict(torch.load("video_cnn_weights.pth", map_location=device))
    video_model.eval()
except FileNotFoundError:
    print("Warning: video_cnn_weights.pth not found.")

video_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. LABELS & HTML TEMPLATE
# ==========================================
CLASS_NAMES = {
    0: {"label": "AI-Generated", "color": "#b30000", "description": "This media appears to be created by artificial intelligence."},
    1: {"label": "Human-Made", "color": "#00802b", "description": "This media appears to be an authentic, real recording."}
}

def render_result(result, confidence_percent, back_url):
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Analysis Result</title>
            <link rel="stylesheet" href="/style.css">
            <style>
                .result-box {
                    border: 3px solid {{ result.color }};
                    border-radius: 12px;
                    padding: 30px;
                    margin: 25px 0 15px;
                    background-color: #faf6ef;
                    text-align: center;
                }
                .result-tag {
                    font-size: 34px; font-weight: bold; color: {{ result.color }}; margin-bottom: 10px;
                }
                .confidence {
                    font-size: 24px; color: #382f2d; margin-top: 15px;
                }
                .disclaimer-box {
                    background-color: #fff3f3; border-left: 5px solid #d9534f;
                    padding: 20px; margin-bottom: 30px; border-radius: 4px; text-align: left;
                }
                .disclaimer-box h3 {
                    color: #d9534f; margin-top: 0; margin-bottom: 10px; font-size: 22px;
                }
                .disclaimer-box p {
                    font-size: 18px; color: #555; margin: 0; line-height: 1.5;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Result</h1>
                <hr class="divider">
                <div class="result-box">
                    <div class="result-tag">{{ result.label }}</div>
                    <p style="font-size: 20px;">{{ result.description }}</p>
                    <p class="confidence">Confidence Score: <strong>{{ confidence_percent }}</strong></p>
                </div>
                <div class="disclaimer-box">
                    <h3>⚠️ Important Disclaimer</h3>
                    <p>
                        <strong>This tool is not 100% accurate.</strong> AI detection technology is experimental and frequently makes mistakes. 
                        <br><br>
                        <strong>Risks of use:</strong> This model may falsely flag a real, human-made recording as AI, or it may completely miss a sophisticated AI-generated file. <strong>Do not use this tool</strong> for legal, academic, or professional decisions.
                    </p>
                </div>
                <a href="{{ back_url }}" class="action-button" style="text-align:center; display:block; text-decoration:none; box-sizing:border-box;">Check Another File</a>
                <a href="/index.html" class="secondary-button" style="text-align:center;">Back to Main Menu</a>
            </div>
        </body>
        </html>
    """, result=result, confidence_percent=confidence_percent, back_url=back_url)

# ==========================================
# 4. WEB ROUTES
# ==========================================
@app.route("/upload-photo", methods=["POST"])
def upload_photo():
    file_list = request.files.getlist("file")
    if not file_list or file_list[0].filename == "":
        return "No selected file", 400
    if len(file_list) > 1:
        return "Please upload only one file at a time.", 400
        
    try:
        image = Image.open(io.BytesIO(file_list[0].read())).convert("RGB")
        tensor = photo_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = photo_model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        result = CLASS_NAMES[predicted_idx.item()]
        confidence_percent = f"{confidence.item() * 100:.1f}%"
        return render_result(result, confidence_percent, "/photo.html")
    except Exception as e:
        return f"Error processing photo: {str(e)}", 500

@app.route("/upload-video", methods=["POST"])
def upload_video():
    file_list = request.files.getlist("file")
    if not file_list or file_list[0].filename == "":
        return "No selected file", 400
    if len(file_list) > 1:
        return "Please upload only one file at a time.", 400

    video_file = file_list[0]
    temp_path = None
    try:
        # OpenCV requires a file path on disk, so we save the upload temporarily
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_file.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        frame_predictions = []
        frame_count = 0
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract 1 frame out of every 15[cite: 3]
                if frame_count % 15 == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    input_tensor = video_transforms(img).unsqueeze(0).to(device)
                    
                    outputs = video_model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    real_prob = probs[0][1].item() # Probability of index 1 (Real)
                    frame_predictions.append(real_prob)
                    
                frame_count += 1
        cap.release()

        if not frame_predictions:
            return "Could not extract any frames from video.", 400

        # Average the probabilities across all frames[cite: 3]
        avg_real_prob = np.mean(frame_predictions)
        avg_fake_prob = 1.0 - avg_real_prob
        
        verdict_idx = 1 if avg_real_prob > 0.5 else 0
        confidence = max(avg_real_prob, avg_fake_prob) * 100
        
        result = CLASS_NAMES[verdict_idx]
        confidence_percent = f"{confidence:.1f}%"
        
        return render_result(result, confidence_percent, "/video.html")
        
    except Exception as e:
        return f"Error processing video: {str(e)}", 500
    finally:
        # Always clean up the temporary video file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/")
@app.route("/<path:filename>")
def serve_static(filename="index.html"):
    from flask import send_from_directory
    return send_from_directory(".", filename)

if __name__ == "__main__":
    print(">>> Starting the web server with Photo and Video support...")
    app.run(debug=True, host="127.0.0.1", port=5000)