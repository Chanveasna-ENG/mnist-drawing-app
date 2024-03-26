import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


mlp_model = MLP(784)
mlp_model.load_state_dict(torch.load('mlp_model.pth'))


# GUI PART
grid_size = 28
pixel_size = 20

pixels = [[0 for _ in range(grid_size)] for _ in range(grid_size)]


def draw(event):
    x = event.x // pixel_size
    y = event.y // pixel_size
    canvas.create_rectangle(x*pixel_size, y*pixel_size, (x+1)*pixel_size, (y+1)*pixel_size, fill="black")
    pixels[y][x] = 1 # black

    # Color the surrounding pixels
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and pixels[ny][nx] == 0:
                canvas.create_rectangle(nx*pixel_size, ny*pixel_size, (nx+1)*pixel_size, (ny+1)*pixel_size, fill="gray")
                pixels[ny][nx] = 0.5  # gray
    predict()


def clear():
    canvas.delete("all")
    for i in range(grid_size):
        for j in range(grid_size):
            pixels[i][j] = 0
    prediction_label.config(text=f"Predicted class: ")


def predict():
    image = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0)
    output = mlp_model(image.view(image.size(0), -1))
    prediction = output.argmax(dim=1).item()
    prediction_label.config(text=f"Predicted class: {prediction}")


root = tk.Tk()

canvas = tk.Canvas(root, width=grid_size*pixel_size,
                   height=grid_size*pixel_size)
canvas.bind("<B1-Motion>", draw)
canvas.pack()

button = tk.Button(root, text="Clear", command=clear)
button.pack()

prediction_label = tk.Label(root, text="")
prediction_label.pack()

root.mainloop()
