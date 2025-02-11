import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import os
import json
from PIL import Image, ImageTk
from src.preprocessing import preprocess_image
from src.detection import detect_steps
from src.evaluation import evaluate_model

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stair Counter")
        self.configure(bg='black')
        self.geometry("800x600")
        self.minsize(400, 300)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='black')
        self.style.configure('TButton', background='black', foreground='white')
        self.style.map('TButton', background=[('active', 'gray20')])
        
        self.create_widgets()
        self.current_image = None
        self.image_paths = []
        self.current_index = 0
        self.predictions = {}
        self.ground_truth = {}
        
        self.bind('<Left>', self.prev_image)
        self.bind('<Right>', self.next_image)
        self.bind('<t>', self.process_image)
        self.bind('<e>', self.evaluate)
        self.bind('<Configure>', self.on_window_resize)

    def create_widgets(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        self.select_btn = ttk.Button(self.main_frame, text="Select Folder (Ctrl+O)", command=self.load_folder)
        self.select_btn.pack(pady=10)
        
        self.load_gt_btn = ttk.Button(self.main_frame, text="Load Ground Truth", command=self.load_ground_truth)
        self.load_gt_btn.pack(pady=10)
        
        self.canvas = tk.Canvas(self.main_frame, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')
        
        self.info_label = ttk.Label(self.main_frame, text="[← →] Naviguer | [T] Traiter | [E] Évaluer | [ESC] Quitter", foreground='white', background='black')
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
            print("Ground truth loaded successfully.")
    
    def show_image(self):
        if self.image_paths:
            img_path = self.image_paths[self.current_index]
            self.current_image = cv2.imread(img_path)
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
    
    def process_image(self, event=None):
        if self.current_image is not None:
            processed = preprocess_image(self.current_image)
            count, debug_img = detect_steps(processed, self.current_image.copy())
            self.current_image = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            self.update_image_display()
            
            img_name = os.path.basename(self.image_paths[self.current_index])
            self.predictions[img_name] = count
            
            canvas_width = self.canvas.winfo_width()
            self.canvas.create_text(10, 10, text=f"Marches détectées: {count}", anchor='nw', fill='white', font=('Courier New', 14, 'bold'))
    
    def evaluate(self):
        if not self.image_paths:
            return
        
        gt_file_path = filedialog.askopenfilename(title="Select Ground Truth JSON", filetypes=[("JSON files", "*.json")])
        if not gt_file_path:
            return
        
        print("Ground truth loaded successfully.")
        
        preds = [self.detected_steps.get(os.path.basename(img), 0) for img in self.image_paths]
        acc, error = evaluate_model(preds, gt_file_path)
        
        print(f"Accuracy: {acc:.2f}, MAE: {error:.2f}")
    
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
