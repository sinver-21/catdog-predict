import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import torchvision.models as models

# Tải model đã train
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("cats_dogs_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return "Cat" if predicted.item() == 0 else "Dog"

def choose_file():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        result = predict_image(file_path)
        messagebox.showinfo("Kết quả dự đoán", f"Ảnh này là: {result}")

# Giao diện
root = tk.Tk()
root.title("Dự đoán Mèo hoặc Chó")
root.geometry("300x150")

btn = tk.Button(root, text="Chọn ảnh để dự đoán", command=choose_file, font=("Arial", 12))
btn.pack(expand=True)

root.mainloop()
