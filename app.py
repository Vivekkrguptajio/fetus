from flask import Flask, render_template, request, session, send_file
import torch
from torchvision import models, transforms
from PIL import Image
import os
import time
import uuid
import matplotlib.pyplot as plt
import gdown  # ✅ For downloading model from Google Drive

app = Flask(__name__)
app.secret_key = 'fetal_secret_key'
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Ensure model folder
os.makedirs("model", exist_ok=True)
model_path = "model/resnet18_fetal.pth"

# ✅ Download from Google Drive if not found
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=1BJT6PjcvcGz_K73StDiZa83s2a2jowty"  # Replace with your Drive ID
    gdown.download(url, model_path, quiet=False)

# ✅ Load model
class_names = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other']
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ✅ Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 🔍 Predict image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
    return class_names[pred_idx.item()], round(confidence.item() * 100, 2), probs.tolist()

# 🏠 Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = str(int(time.time())) + '_' + file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            label, confidence, probs = predict_image(path)
            session['history'].append({
                "image": filename,
                "label": label,
                "confidence": confidence
            })

            # 📊 Bar chart
            plt.figure(figsize=(8, 4))
            plt.bar(class_names, probs, color='skyblue')
            plt.xticks(rotation=45)
            plt.ylabel("Confidence")
            plt.tight_layout()
            chart_path = f"static/chart_{uuid.uuid4().hex}.png"
            plt.savefig(chart_path)
            plt.close()

            return render_template('index.html', filename=filename, label=label, confidence=confidence,
                                   chart_path=chart_path, history=session['history'])
    return render_template('index.html', history=session.get('history', []))

# 📄 Downloadable report
@app.route('/download_report')
def download_report():
    if 'history' not in session or not session['history']:
        return "No predictions yet."

    report_text = "Fetal Ultrasound Prediction Report\n\n"
    for entry in session['history']:
        report_text += f"Image: {entry['image']}\nPrediction: {entry['label']}\nConfidence: {entry['confidence']}%\n\n"

    report_path = "static/report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    return send_file(report_path, as_attachment=True)

# ✅ Run app
if __name__ == '__main__':
    app.run(debug=True)
