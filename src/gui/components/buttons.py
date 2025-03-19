import tkinter as tk
from tkinter import ttk

class Buttons:
    def __init__(self, parent, controller):
        """
        Initialiser le composant des boutons.

        Args:
            parent: Le widget parent (par exemple, un cadre).
            controller: Le contrôleur principal de l'application (instance de Interface).
        """
        self.parent = parent
        self.controller = controller  # Référence à la classe principale Interface

        # Créer le cadre des boutons
        self.button_frame = ttk.Frame(self.parent)
        self.button_frame.pack(fill='x', pady=10)

        # Bouton pour charger un dossier d'images
        self.select_btn = ttk.Button(self.button_frame, text="Choisir un dossier (Ctrl+o)", command=self.controller.load_folder)
        self.select_btn.pack(side='left', padx=5)

        # Bouton pour charger les données de vérité terrain
        self.load_gt_btn = ttk.Button(self.button_frame, text="Charger la vérité terrain (Ctrl+g)", command=self.controller.load_ground_truth)
        self.load_gt_btn.pack(side='left', padx=5)

        # Bouton pour évaluer toutes les images
        self.evaluate_btn = ttk.Button(self.button_frame, text="Évaluer le set (Ctrl+e)", command=self.controller.evaluate_all_images)
        self.evaluate_btn.pack(side='left', padx=5)