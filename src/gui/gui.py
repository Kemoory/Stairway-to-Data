import tkinter as tk
from tkinter import ttk, filedialog
from .components.model_selection import ModelSelection
from .components.preprocessing_selection import PreprocessingSelection
from .components.buttons import Buttons
from .components.image_display import ImageDisplay
from src.gui.utils import preprocess_image, detect_steps
import cv2
import os
import json
from tkinter import filedialog
from ..evaluation.evaluation import evaluate_all_combinations

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compteur de marches")
        self.configure(bg='#2E3440')
        self.geometry("1000x700")
        self.minsize(800, 600)

        # Initialiser les variables (le point de départ de notre ascension)
        self.processed_image = None
        self.debug_image = None
        self.current_image = None
        self.original_image = None
        self.image_paths = []
        self.current_index = 0
        self.predictions = {}
        self.ground_truth = {}

        # Configurer les styles (pour que l'interface soit au sommet de son art)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2E3440')
        self.style.configure('TButton', background='#4C566A', foreground='#ECEFF4', font=('Helvetica', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#5E81AC')])
        self.style.configure('TLabel', background='#2E3440', foreground='#ECEFF4', font=('Helvetica', 10))
        self.style.configure('Status.TLabel', background='#2E3440', foreground='#ECEFF4', font=('Helvetica', 9, 'italic'))
        self.style.configure('TCombobox', background='#4C566A', foreground='#403A2E', font=('Helvetica', 10))

        # Créer la disposition principale (le plan de l'escalier)
        self.create_widgets()

        # Associer les événements clavier (pour monter ou descendre rapidement)
        self.bind('<Left>', self.prev_image)
        self.bind('<Right>', self.next_image)
        self.bind('<t>', self.process_image)
        self.bind('<Configure>', self.on_window_resize)
        self.bind('<Control-o>', self.load_folder)  # Ctrl+O pour charger un dossier
        self.bind('<Control-g>', self.load_ground_truth)  # Ctrl+G pour charger la vérité terrain
        self.bind('<Control-e>', self.evaluate_all_images)  # Ctrl+E pour évaluer toutes les images
        self.bind('<Control-d>', self.load_from_db)  # Ajouter cette ligne  # Ctrl+D pour charger les données depuis la base de données

    def create_widgets(self):
        """Créer la disposition principale et charger les composants."""
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Charger les composants (les marches de l'interface)
        self.model_selection = ModelSelection(self.main_frame, self)
        self.preprocessing = PreprocessingSelection(self.main_frame, self)
        self.buttons = Buttons(self.main_frame, self)
        self.image_display = ImageDisplay(self.main_frame, self)

    def load_from_db(self, event=None):
        """Charger les données depuis PostgreSQL"""
        try:
            import psycopg2
            from src.config import db_config
            
            # Connexion à la base de données
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # Exécution de la requête
            cursor.execute("""
                SELECT image_path, nombre_de_marches 
                FROM public.images_data
                WHERE image_path IS NOT NULL
            """)
            
            # Récupération des résultats
            results = cursor.fetchall()
            
            # Mise à jour des données
            self.image_paths = []
            self.ground_truth = {}
            
            for row in results:
                image_path = row[0].replace('./', '')  # Ajustement du chemin
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.ground_truth[os.path.basename(image_path)] = row[1]
            
            if self.image_paths:
                self.current_index = 0
                self.show_image()
                self.image_display.info_label.config(text=f"Base de données chargée : {len(self.image_paths)} images trouvées")
            else:
                self.image_display.info_label.config(text="Aucune image valide trouvée dans la base de données")
                
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.image_display.info_label.config(text=f"Erreur DB : {str(e)}")

    def load_folder(self, event=None):
        """Charger un dossier d'images (le point de départ de notre escalade)."""
        folder_path = filedialog.askdirectory(initialdir='data/raw')
        if folder_path:
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if self.image_paths:
                self.current_index = 0
                self.show_image()

    def load_ground_truth(self, event=None):
        """Charger les données de vérité terrain depuis un fichier JSON (pour ne pas perdre pied)."""
        gt_path = filedialog.askopenfilename(initialdir='data', filetypes=[("JSON files", "*.json")])
        if gt_path:
            with open(gt_path, 'r') as f:
                data = json.load(f)
            self.ground_truth = {item["Images"]: item["Nombre de marches"] for item in data}
            self.image_display.info_label.config(text="Vérité terrain chargée avec succès.")

    def show_image(self):
        """Afficher l'image actuelle (pour ne pas manquer une marche)."""
        if self.image_paths:
            img_path = self.image_paths[self.current_index]
            self.original_image = cv2.imread(img_path)
            self.current_image = self.original_image.copy()
            self.image_display.update_image_display(self.current_image)

    def reset_image(self, event=None):
        """Réinitialiser l'image à son état d'origine (retour au rez-de-chaussée)."""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.image_display.update_image_display(self.current_image)

    def process_image(self, event=None):
        """Traiter l'image actuelle avec le prétraitement et le modèle sélectionnés (pour gravir les marches)."""
        if self.current_image is not None:
            # Obtenir la méthode de prétraitement sélectionnée
            preprocessing_method = self.preprocessing.preprocess_var.get()

            # Prétraiter l'image
            processed = preprocess_image(self.current_image, preprocessing_method)

            # Obtenir la méthode de détection des marches sélectionnée
            model_method = self.model_selection.model_var.get()

            # Détecter les marches dans l'image prétraitée
            count, debug_img = detect_steps(processed, model_method, self.current_image)

            # Mettre à jour les images traitées et de débogage
            self.processed_image = processed
            self.debug_image = debug_img

            # Mettre à jour l'affichage de l'image
            self.image_display.update_image_display(self.current_image, self.processed_image, self.debug_image)

            # Sauvegarder la prédiction
            img_name = os.path.basename(self.image_paths[self.current_index])
            self.predictions[img_name] = count

            # Afficher le nombre de marches détectées
            self.image_display.canvas.create_text(10, 10, text=f"Marches détectées: {count}", anchor='nw', fill='white', font=('Helvetica', 14, 'bold'))

    def evaluate_all_images(self, event=None):
        """Évaluer toutes les images du dossier chargé (pour atteindre le sommet de l'évaluation)."""
        if not self.image_paths:
            self.image_display.info_label.config(text="Aucune image chargée pour l'évaluation.")
            return
        if not self.ground_truth:
            self.image_display.info_label.config(text="Vérité terrain non chargée. Veuillez la charger d'abord.")
            return

        # Évaluer toutes les combinaisons
        results = evaluate_all_combinations(self.image_paths, self.ground_truth)
        self.image_display.info_label.config(text="Évaluation terminée. Résultats sauvegardés dans 'evaluation_results.json'.")

    def next_image(self, event=None):
        """Passer à l'image suivante (une marche à la fois)."""
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_image()

    def prev_image(self, event=None):
        """Revenir à l'image précédente (ne pas descendre trop vite)."""
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_image()

    def on_window_resize(self, event=None):
        """Gérer le redimensionnement de la fenêtre (pour ne pas perdre l'équilibre)."""
        if self.current_image is not None:
            self.image_display.update_image_display(self.current_image, self.processed_image, self.debug_image)