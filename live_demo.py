import torch
import clip
import cv2
from PIL import Image

PRED_THRESHOLD = 0.8

# CLIP setting
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Target(text labels) settings
targets = ["iPhone", "man", "woman", "coffie", "papers"]
text = clip.tokenize(targets).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

    # Open the camera
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is not True:
            break

        # Get image feature
        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)

        # Map image and text feature to prediction
        logits_per_image, logits_per_text = model(image, text)
    
        predict_index = logits_per_image.cpu().numpy().argmax()
        target = targets[predict_index]
        prob = logits_per_image.softmax(dim=-1).cpu().numpy()[0, predict_index]
        
        if prob > PRED_THRESHOLD:
            frame = cv2.putText(frame, f'{target}: {prob}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

        # print image
        cv2.imshow('sample', frame)
        cv2.waitKey(1)
