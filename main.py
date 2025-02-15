import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import os
import json
from PIL import Image, ImageTk
from src.preprocessing import preprocess_image, preprocess_image_alternative, split, merge
from src.model.detection import detect_steps, detect_steps_alternative, detect_vanishing_lines, fourier_transform
from src.evaluation import evaluate_model

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stair Counter")
        self.configure(bg='#2E3440')
        self.geometry("1000x700")
        self.minsize(800, 600)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2E3440')
        self.style.configure('TButton', background='#4C566A', foreground='#ECEFF4', font=('Helvetica', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#5E81AC')])
        self.style.configure('TLabel', background='#2E3440', foreground='#ECEFF4', font=('Helvetica', 10))
        self.style.configure('Status.TLabel', background='#2E3440', foreground='#ECEFF4', font=('Helvetica', 9, 'italic'))
        self.style.configure('TCombobox', background='#4C566A', foreground='#403A2E', font=('Helvetica', 10))

        self.create_widgets()
        self.current_image = None
        self.original_image = None 
        self.image_paths = []
        self.current_index = 0
        self.predictions = {}
        self.ground_truth = {}

        self.bind('<Left>', self.prev_image)
        self.bind('<Right>', self.next_image)
        self.bind('<t>', self.process_image)
        self.bind('<Configure>', self.on_window_resize)

    def create_widgets(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Dropdown for model selection
        self.model_frame = ttk.Frame(self.main_frame)
        self.model_frame.pack(fill='x', pady=5)
        self.model_label = ttk.Label(self.model_frame, text="Select Model:")
        self.model_label.pack(side='left', padx=5)
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, state='readonly')
        self.model_combobox['values'] = (
            'HoughLinesP', 
            'HoughLinesP Alternative', 
            'Vanishing Lines', 
            'Fourier Transform'
        )
        self.model_combobox.current(0)  # Selection par defaut
        self.model_combobox.pack(side='left', padx=5)
        self.model_combobox.bind('<<ComboboxSelected>>', self.reset_image)  # Reset l'image lors du changement de model

        # Dropdown pour le preprocessing
        self.preprocess_frame = ttk.Frame(self.main_frame)
        self.preprocess_frame.pack(fill='x', pady=5)
        self.preprocess_label = ttk.Label(self.preprocess_frame, text="Select Preprocessing:")
        self.preprocess_label.pack(side='left', padx=5)
        self.preprocess_var = tk.StringVar()
        self.preprocess_combobox = ttk.Combobox(self.preprocess_frame, textvariable=self.preprocess_var, state='readonly')
        self.preprocess_combobox['values'] = (
            'Gaussian Blur + Canny', 
            'Median Blur + Canny', 
            'Split and Merge'
        )
        self.preprocess_combobox.current(0)  # Selection par defaut
        self.preprocess_combobox.pack(side='left', padx=5)
        self.preprocess_combobox.bind('<<ComboboxSelected>>', self.reset_image)  # Reset l'image lors du changement de preprocessing

        # Buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill='x', pady=10)

        self.select_btn = ttk.Button(self.button_frame, text="Select Folder (Ctrl+O)", command=self.load_folder)
        self.select_btn.pack(side='left', padx=5)

        self.load_gt_btn = ttk.Button(self.button_frame, text="Load Ground Truth", command=self.load_ground_truth)
        self.load_gt_btn.pack(side='left', padx=5)

        self.evaluate_btn = ttk.Button(self.button_frame, text="Evaluate Set", command=self.evaluate_all_images)
        self.evaluate_btn.pack(side='left', padx=5)

        # Canvas
        self.canvas = tk.Canvas(self.main_frame, bg='#3B4252', highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')

        # Label
        self.info_label = ttk.Label(self.main_frame, text="[← →] Navigate | [T] Process | [ESC] Quit (TODO)", style='Status.TLabel')
        self.info_label.pack(pady=10)

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if self.image_paths:
                self.current_index = 0
                self.show_image()

    def load_ground_truth(self):
        gt_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if gt_path:
            with open(gt_path, 'r') as f:
                data = json.load(f)
            self.ground_truth = {item["Images"]: item["Nombre de marches"] for item in data}
            self.info_label.config(text="Ground truth loaded successfully.")

    def show_image(self):
        if self.image_paths:
            img_path = self.image_paths[self.current_index]
            self.original_image = cv2.imread(img_path)  # Store l'image original
            self.current_image = self.original_image.copy()  # Reset pour l'image original
            self.update_image_display()

    def update_image_display(self):
        if self.current_image is not None:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            h, w = self.current_image.shape[:2]
            ratio = min(canvas_width/w, canvas_height/h)
            new_size = (int(w*ratio), int(h*ratio))

            img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize(new_size)
            self.tk_img = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, anchor='center', image=self.tk_img)
            self.title(f"Stair Counter - {os.path.basename(self.image_paths[self.current_index])}")

    def reset_image(self, event=None):
        """Reset the image to the original unprocessed version"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_image_display()

    def process_image(self, event=None):
        if self.current_image is not None:
            # Obtenir la methode de pretraitement selectionnee
            preprocessing_method = self.preprocess_var.get()
            if preprocessing_method == 'Gaussian Blur + Canny':
                processed = preprocess_image(self.current_image)
            elif preprocessing_method == 'Median Blur + Canny':
                processed = preprocess_image_alternative(self.current_image)
            elif preprocessing_method == 'Split and Merge':
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                regions = split(gray, threshold=10)
                merged = merge(gray, regions, threshold=10)
                processed = cv2.Canny(merged, 50, 150)  # Convertir en image binaire des contours

            # Obtenir le modèle sélectionné
            model_method = self.model_var.get()
            if model_method == 'HoughLinesP':
                count, debug_img = detect_steps(processed, self.current_image.copy())
            elif model_method == 'HoughLinesP Alternative':
                count, debug_img = detect_steps_alternative(processed, self.current_image.copy())
            elif model_method == 'Vanishing Lines':
                debug_img, count = detect_vanishing_lines(self.current_image.copy())
            elif model_method == 'Fourier Transform':
                debug_img = fourier_transform(self.current_image.copy())
                count = 0  # Pas de comptage de marches pour la transformée de Fourier

            self.current_image = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            self.update_image_display()

            img_name = os.path.basename(self.image_paths[self.current_index])
            self.predictions[img_name] = count

            canvas_width = self.canvas.winfo_width()
            self.canvas.create_text(10, 10, text=f"Marches détectées: {count}", anchor='nw', fill='white', font=('Helvetica', 14, 'bold'))

    def evaluate_all_images(self, event=None):
        if not self.image_paths:
            self.info_label.config(text="No images loaded for evaluation.")
            return
        if not self.ground_truth:
            self.info_label.config(text="Ground truth not loaded. Please load it first.")
            return
        preds = {}
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            # Obtenir la methode de pretraitement selectionnee
            preprocessing_method = self.preprocess_var.get()
            if preprocessing_method == 'Gaussian Blur + Canny':
                processed = preprocess_image(img)
            elif preprocessing_method == 'Median Blur + Canny':
                processed = preprocess_image_alternative(img)

            # Obtenir le modèle sélectionné
            model_method = self.model_var.get()
            if model_method == 'HoughLinesP':
                count, _ = detect_steps(processed, img.copy())
            elif model_method == 'HoughLinesP Alternative':
                count, _ = detect_steps_alternative(processed, img.copy())

            preds[os.path.basename(img_path)] = count

        error, rel_error, precision, recall, conf_matrix = evaluate_model(preds, self.ground_truth)

        self.info_label.config(text=f"Evaluation Results: MAE: {error:.2f}, Relative Error: {rel_error:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")

    def next_image(self, event=None):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_image()

    def prev_image(self, event=None):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_image()

    def on_window_resize(self, event=None):
        if self.current_image is not None:
            self.update_image_display()

if __name__ == "__main__":
    app = Interface()
    app.mainloop()