import os
import tkinter as tk
from tkinter import Radiobutton, Label, Entry, Button, messagebox
import json
import main

# Загрузка переводов из файла
trs = {}
for file in os.listdir():
    if file.endswith(".lang"):
        with open(file, 'r', encoding='utf-8') as f:
            trs = json.load(f)

def create_label_and_entry(root, text):
    Label(root, text=trs[text]).pack(pady=10)
    entry = Entry(root)
    entry.pack()
    return entry

def get_user_input():
    root = tk.Tk()
    root.geometry("380x500")
    root.title(trs["training_settings"])
    root.resizable(False, False)

    n_epochs_entry = create_label_and_entry(root, "enter_epochs")
    batch_size_entry = create_label_and_entry(root, "enter_batch_size")
    image_size_x_entry = create_label_and_entry(root, "enter_image_size_x")
    image_size_y_entry = create_label_and_entry(root, "enter_image_size_y")

    device_choice = tk.StringVar(value="cpu")
    Label(root, text=trs["choose_device"]).pack(pady=10)
    Radiobutton(root, text="CPU", variable=device_choice, value="cpu").pack()
    Radiobutton(root, text="GPU", variable=device_choice, value="gpu").pack()

    start_choice = tk.StringVar(value="new")
    Label(root, text=trs["choose_start_mode"]).pack(pady=10)
    Radiobutton(root, text=trs["new_training"], variable=start_choice, value="new").pack()
    Radiobutton(root, text=trs["resume_training"], variable=start_choice, value="resume").pack()

    def submit():
        global n_epochs, batch_size, image_size_x, image_size_y, device_choice_value, start_choice_value
        n_epochs = int(n_epochs_entry.get())
        batch_size = int(batch_size_entry.get())
        image_size_x = int(image_size_x_entry.get())
        image_size_y = int(image_size_y_entry.get())
        device_choice_value = device_choice.get()
        start_choice_value = start_choice.get()
        dataset_dir = 'dataset'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            messagebox.showinfo(trs["info"], trs["dataset_not_found"])
        elif not os.listdir(dataset_dir):
            messagebox.showinfo(trs["info"], trs["dataset_empty"])
        else:
            root.quit()

    Button(root, text=trs["start"], command=submit, font=("Arial", 20), width=20, height=3).pack(pady=10)

    root.mainloop()

    return n_epochs, batch_size, image_size_x, image_size_y, device_choice_value, start_choice_value

def training_completed():
    root = tk.Tk()
    root.geometry("200x100")
    root.resizable(False, False)
    root.title(trs["training_completed"])
    Label(root, text=trs["training_completed"]).pack(pady=10)
    Button(root, text="ОК", command=root.quit).pack(pady=10)
    root.mainloop()
