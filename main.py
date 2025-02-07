# main.py
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import os
from PIL import Image, ImageTk
from src.preprocessing import preprocess_image
from src.detection import detect_steps
#from src.evaluation import evaluate_model Pas encore implementer

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stair Counter")
        self.configure(bg='black')
        self.geometry("800x600")
        self.minsize(400, 300)  #Taille min de la fenetre
        
        #Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='black')
        self.style.configure('TButton', 
                           background='black', 
                           foreground='white',
                           bordercolor='white',
                           lightcolor='black',
                           darkcolor='black')
        self.style.map('TButton', 
                      background=[('active', 'gray20')])
        
        #Interface
        self.create_widgets()
        self.current_image = None
        self.image_paths = []
        self.current_index = 0
        
        #Bind des touches
        self.bind('<Left>', self.prev_image)
        self.bind('<Right>', self.next_image)
        self.bind('<t>', self.process_image)
        self.bind('<Configure>', self.on_window_resize)  #Gere le redimensionnement de la fenêtre

    def create_widgets(self):
        #Frame principale
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        #(Top) Bouton
        self.select_btn = ttk.Button(self.main_frame, 
                                   text="Select Folder (Ctrl+O)", 
                                   command=self.load_folder)
        self.select_btn.pack(pady=10)
        
        #(Mid) Zone d'affichage image
        self.canvas = tk.Canvas(self.main_frame, 
                              bg='black', 
                              highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')
        
        #(Bottom) Panneau d'information
        self.info_label = ttk.Label(self.main_frame, 
                                  text="[← →] Naviguer | [SPACE] Traiter | [ESC] Quitter",
                                  foreground='white',
                                  background='black')
        self.info_label.pack(pady=10)
        
    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_paths = [os.path.join(folder_path, f) 
                              for f in os.listdir(folder_path) 
                              if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if self.image_paths:
                self.current_index = 0
                self.show_image()

    def show_image(self):
        if self.image_paths:
            img_path = self.image_paths[self.current_index]
            self.current_image = cv2.imread(img_path)
            self.update_image_display()

    def update_image_display(self):
        if self.current_image is not None:
            #Redimensionnement proportionnel basé sur la taille actuelle de la fenêtre
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
            
            #Résultat
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            self.current_image = debug_img
            self.update_image_display()
            
            #Affichage du nombre de marches
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_text(10, 10, 
                                  text=f"Marches détectées: {count}", 
                                  anchor='nw', 
                                  fill='white',
                                  font=('Courier New', 14, 'bold'))

    def next_image(self, event=None):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_image()

    def prev_image(self, event=None):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_image()

    def on_window_resize(self, event=None):
        #Maj de l'affichage de l'image lors du redimensionnement
        if self.current_image is not None:
            self.update_image_display()

if __name__ == "__main__":
    app = Interface()
    app.mainloop()