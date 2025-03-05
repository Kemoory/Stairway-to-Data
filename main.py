#main.py
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import os
import json
import numpy as np
from PIL import Image, ImageTk

from src.preprocessing.gaussian import preprocess_gaussian
from src.preprocessing.median import preprocess_median
from src.preprocessing.splitAndMerge import preprocess_splitAndMerge
from src.preprocessing.adaptive_tresholding import preprocess_adaptive_thresholding
from src.preprocessing.gradient_orientation import preprocess_gradient_orientation
from src.preprocessing.homorphic_filter import preprocess_homomorphic_filter
from src.preprocessing.phase_congruency import preprocess_phase_congruency
from src.preprocessing.wavelet import preprocess_image_wavelet

from src.model.houghLineSeg import detect_steps_houghLineSeg
from src.model.houghLineExt import detect_steps_houghLineExt
from src.model.RANSAC import detect_steps_RANSAC
from src.model.vanishingLine import detect_vanishing_lines

from src.evaluation import evaluate_all_combinations

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compteur de marches")  # Parce qu'on compte les marches, pas les étoiles
        self.configure(bg='#2E3440')  # Fond sombre pour un style pro
        self.geometry("1000x700")  # Taille de la fenêtre
        self.minsize(800, 600)  # Taille minimale, parce qu'on est pas des sauvages

        self.processed_image = None  # Image traitée
        self.debug_image = None  # Image de débogage (avec les lignes détectées)

        self.style = ttk.Style()
        self.style.theme_use('clam')  # Thème "clam" pour un look moderne
        self.style.configure('TFrame', background='#2E3440')  # Fond du frame
        self.style.configure('TButton', background='#4C566A', foreground='#ECEFF4', font=('Helvetica', 10, 'bold'))  # Boutons stylés
        self.style.map('TButton', background=[('active', '#5E81AC')])  # Boutons qui changent de couleur quand on clique
        self.style.configure('TLabel', background='#2E3440', foreground='#ECEFF4', font=('Helvetica', 10))  # Labels stylés
        self.style.configure('Status.TLabel', background='#2E3440', foreground='#ECEFF4', font=('Helvetica', 9, 'italic'))  # Label de statut
        self.style.configure('TCombobox', background='#4C566A', foreground='#403A2E', font=('Helvetica', 10))  # Combobox stylé

        self.create_widgets()  # Crée les widgets de l'interface
        self.current_image = None  # Image actuelle
        self.original_image = None  # Image originale (avant traitement)
        self.image_paths = []  # Liste des chemins des images
        self.current_index = 0  # Index de l'image actuelle
        self.predictions = {}  # Prédictions du modèle
        self.ground_truth = {}  # Vérité terrain (pour l'évaluation)

        self.bind('<Left>', self.prev_image)  # Flèche gauche pour l'image précédente
        self.bind('<Right>', self.next_image)  # Flèche droite pour l'image suivante
        self.bind('<t>', self.process_image)  # Touche 'T' pour traiter l'image
        self.bind('<Configure>', self.on_window_resize)  # Redimensionnement de la fenêtre

    def create_widgets(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)  # Frame principal

        # Dropdown pour la sélection du modèle
        self.model_frame = ttk.Frame(self.main_frame)
        self.model_frame.pack(fill='x', pady=5)
        self.model_label = ttk.Label(self.model_frame, text="Choix du modèle :")  # Label pour le modèle
        self.model_label.pack(side='left', padx=5)
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, state='readonly')  # Combobox pour choisir le modèle
        self.model_combobox['values'] = (
            'HoughLinesP (Segmented)',  # Modèle 1 (par defaut)
            'HoughLinesP (Extended)',  # Modèle 2 (cherche des pattern de recursivité)
            'Vanishing Lines',  # Modèle 3 (peut etre opti)
            'RANSAC (WIP)',  # Modèle 4 (pour les fans de maths)
        )
        self.model_combobox.current(0)  # Sélection par défaut
        self.model_combobox.pack(side='left', padx=5)
        self.model_combobox.bind('<<ComboboxSelected>>', self.reset_image)  # Reset l'image quand on change de modèle

        # Dropdown pour le prétraitement
        self.preprocess_frame = ttk.Frame(self.main_frame)
        self.preprocess_frame.pack(fill='x', pady=5)
        self.preprocess_label = ttk.Label(self.preprocess_frame, text="Choix du prétraitement :")  # Label pour le prétraitement
        self.preprocess_label.pack(side='left', padx=5)
        self.preprocess_var = tk.StringVar()
        self.preprocess_combobox = ttk.Combobox(self.preprocess_frame, textvariable=self.preprocess_var, state='readonly')  # Combobox pour choisir le prétraitement
        self.preprocess_combobox['values'] = (
            '(None)',  # Pas de prétraitement
            'Gaussian Blur + Canny',  # Prétraitement 1
            'Median Blur + Canny',  # Prétraitement 2
            'Split and Merge',  # Prétraitement 3
            'Adaptive Thresholding',  # Prétraitement 4
            'Gradient Orientation',  # Prétraitement 5
            'Homomorphic Filter',  # Prétraitement 6
            'Phase Congruency',  # Prétraitement 7
            'Wavelet Transform',  # Prétraitement 8
        )
        self.preprocess_combobox.current(0)  # Sélection par défaut
        self.preprocess_combobox.pack(side='left', padx=5)
        self.preprocess_combobox.bind('<<ComboboxSelected>>', self.reset_image)  # Reset l'image quand on change de prétraitement

        # Boutons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill='x', pady=10)

        self.select_btn = ttk.Button(self.button_frame, text="Choisir un dossier (Ctrl+O)", command=self.load_folder)  # Bouton pour choisir un dossier
        self.select_btn.pack(side='left', padx=5)

        self.load_gt_btn = ttk.Button(self.button_frame, text="Charger la vérité terrain", command=self.load_ground_truth)  # Bouton pour charger la vérité terrain
        self.load_gt_btn.pack(side='left', padx=5)

        self.evaluate_btn = ttk.Button(self.button_frame, text="Évaluer le set", command=self.evaluate_all_images)  # Bouton pour évaluer toutes les images
        self.evaluate_btn.pack(side='left', padx=5)

        # Canvas pour afficher les images
        self.canvas = tk.Canvas(self.main_frame, bg='#3B4252', highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')

        # Label d'info
        self.info_label = ttk.Label(self.main_frame, text="[← →] Naviguer | [T] Traiter | [ESC] Quitter (TODO)", style='Status.TLabel')  # Infos utiles
        self.info_label.pack(pady=10)

    def load_folder(self):
        folder_path = filedialog.askdirectory(initialdir='data/raw')  # Ouvre une boîte de dialogue pour choisir un dossier
        if folder_path:
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]  # Liste des images dans le dossier
            if self.image_paths:
                self.current_index = 0
                self.show_image()  # Affiche la première image

    def load_ground_truth(self):
        gt_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])  # Ouvre une boîte de dialogue pour charger un fichier JSON
        if gt_path:
            with open(gt_path, 'r') as f:
                data = json.load(f)
            self.ground_truth = {item["Images"]: item["Nombre de marches"] for item in data}  # Charge la vérité terrain
            self.info_label.config(text="Vérité terrain chargée avec succès.")  # Confirmation

    def show_image(self):
        if self.image_paths:
            img_path = self.image_paths[self.current_index]
            self.original_image = cv2.imread(img_path)  # Charge l'image originale
            self.current_image = self.original_image.copy()  # Reset à l'image originale
            # Reset les images traitées et de débogage
            if hasattr(self, 'processed_image'):
                del self.processed_image
            if hasattr(self, 'debug_image'):
                del self.debug_image
            self.update_image_display()  # Met à jour l'affichage

    def update_image_display(self):
        if self.current_image is not None:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Efface le canvas
            self.canvas.delete("all")

            # Cas 1 : Aucun traitement n'a été effectué (montrer uniquement l'image originale)
            if not hasattr(self, 'processed_image') or not hasattr(self, 'debug_image'):
                img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)  # Convertit en RGB
                img = Image.fromarray(img)
                # Récupère les dimensions de l'image
                img_width, img_height = img.size

                # Calcule le ratio d'aspect de l'image et du canvas
                img_aspect_ratio = img_width / img_height
                canvas_aspect_ratio = canvas_width / canvas_height

                # Redimensionne l'image pour qu'elle rentre dans le canvas tout en gardant son ratio
                if img_aspect_ratio > canvas_aspect_ratio:
                    # L'image est plus large que le canvas
                    new_width = canvas_width
                    new_height = int(canvas_width / img_aspect_ratio)
                else:
                    # L'image est plus haute que le canvas
                    new_height = canvas_height
                    new_width = int(canvas_height * img_aspect_ratio)

                # Redimensionne l'image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Calcule la position pour centrer l'image sur le canvas
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2

                # Convertit en PhotoImage et affiche sur le canvas
                self.tk_img = ImageTk.PhotoImage(img)
                self.canvas.create_image(x_offset, y_offset, anchor='nw', image=self.tk_img)

            # Cas 2 : Le traitement a été effectué (montrer l'image traitée et l'image de débogage côte à côte)
            else:
                # Calcule la largeur pour chaque image (moitié du canvas)
                img_width = canvas_width // 2
                img_height = canvas_height

                # Redimensionne et affiche l'image traitée à gauche
                processed_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                processed_img = Image.fromarray(processed_img)

                # Redimensionne l'image traitée pour qu'elle rentre dans la moitié gauche du canvas
                processed_img = processed_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.tk_processed_img = ImageTk.PhotoImage(processed_img)
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_processed_img)

                # Redimensionne et affiche l'image de débogage à droite
                debug_img = cv2.cvtColor(self.debug_image, cv2.COLOR_BGR2RGB)
                debug_img = Image.fromarray(debug_img)

                # Redimensionne l'image de débogage pour qu'elle rentre dans la moitié droite du canvas
                debug_img = debug_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.tk_debug_img = ImageTk.PhotoImage(debug_img)
                self.canvas.create_image(img_width, 0, anchor='nw', image=self.tk_debug_img)

            self.title(f"Compteur de marches - {os.path.basename(self.image_paths[self.current_index])}")  # Met à jour le titre de la fenêtre

    def reset_image(self, event=None):
        """Reset l'image à sa version originale non traitée"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            # Reset les images traitées et de débogage
            if hasattr(self, 'processed_image'):
                del self.processed_image
            if hasattr(self, 'debug_image'):
                del self.debug_image
            self.update_image_display()  # Met à jour l'affichage

    def process_image(self, event=None):
        if self.current_image is not None:
            # Récupère la méthode de prétraitement sélectionnée
            preprocessing_method = self.preprocess_var.get()
            if preprocessing_method == 'Gaussian Blur + Canny':
                processed = preprocess_gaussian(self.current_image)
            elif preprocessing_method == 'Median Blur + Canny':
                processed = preprocess_median(self.current_image)
            elif preprocessing_method == 'Split and Merge':
                processed = preprocess_splitAndMerge(self.current_image)
            elif preprocessing_method == 'Adaptive Thresholding':
                processed = preprocess_adaptive_thresholding(self.current_image)
            elif preprocessing_method == 'Gradient Orientation':
                processed = preprocess_gradient_orientation(self.current_image)
            elif preprocessing_method == 'Homomorphic Filter':
                processed = preprocess_homomorphic_filter(self.current_image)
            elif preprocessing_method == 'Phase Congruency':
                processed = preprocess_phase_congruency(self.current_image)
            elif preprocessing_method == 'Wavelet Transform':
                processed = preprocess_image_wavelet(self.current_image)
            else:
                processed = self.current_image.copy()  # Pas de prétraitement

            # Assure que l'image est en niveaux de gris et en uint8
            if len(processed.shape) > 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            if processed.dtype != np.uint8:
                processed = cv2.convertScaleAbs(processed)

            # Récupère le modèle sélectionné
            model_method = self.model_var.get()
            if model_method == 'HoughLinesP (Segmented)':
                count, debug_img = detect_steps_houghLineSeg(processed, self.current_image.copy())
            elif model_method == 'HoughLinesP (Extended)':
                count, debug_img = detect_steps_houghLineExt(processed, self.current_image.copy())
            elif model_method == 'Vanishing Lines':
                count, debug_img = detect_vanishing_lines(processed, self.current_image.copy())
            elif model_method == 'RANSAC (WIP)':
                count, debug_img = detect_steps_RANSAC(processed, self.current_image.copy())

            self.processed_image = processed  # Stocke l'image traitée
            self.debug_image = debug_img  # Stocke l'image de débogage

            self.update_image_display()  # Met à jour l'affichage

            img_name = os.path.basename(self.image_paths[self.current_index])
            self.predictions[img_name] = count  # Stocke la prédiction

            canvas_width = self.canvas.winfo_width()
            self.canvas.create_text(10, 10, text=f"Marches détectées: {count}", anchor='nw', fill='white', font=('Helvetica', 14, 'bold'))  # Affiche le nombre de marches détectées

    def evaluate_all_images(self, event=None):
        if not self.image_paths:
            self.info_label.config(text="Aucune image chargée pour l'évaluation.")  # Pas d'image, pas d'évaluation
            return
        if not self.ground_truth:
            self.info_label.config(text="Vérité terrain non chargée. Veuillez la charger d'abord.")  # Pas de vérité terrain, pas d'évaluation
            return
        
        # Evaluate all combinations
        results = evaluate_all_combinations(self.image_paths, self.ground_truth)
        
        # Display a summary of the results
        self.info_label.config(text="Évaluation terminée. Résultats sauvegardés dans 'evaluation_results.json'.")

    def next_image(self, event=None):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)  # Passe à l'image suivante
            self.show_image()

    def prev_image(self, event=None):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)  # Passe à l'image précédente
            self.show_image()

    def on_window_resize(self, event=None):
        if self.current_image is not None:
            self.update_image_display()  # Met à jour l'affichage quand on redimensionne la fenêtre

if __name__ == "__main__":
    app = Interface()
    app.mainloop()  # Lance l'application