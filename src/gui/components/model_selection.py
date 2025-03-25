import tkinter as tk
from tkinter import ttk

class ModelSelection:
    def __init__(self, parent, controller):
        """
        Initialiser le composant de sélection de modèle.

        Args:
            parent: Le widget parent (par exemple, un cadre).
            controller: Le contrôleur principal de l'application (instance de Interface).
        """
        self.parent = parent
        self.controller = controller  # Référence à la classe principale Interface

        # Créer le cadre de sélection de modèle
        self.model_frame = ttk.Frame(self.parent)
        self.model_frame.pack(fill='x', pady=5)

        # Étiquette pour le menu déroulant de sélection de modèle
        self.model_label = ttk.Label(self.model_frame, text="Choix du modèle :")
        self.model_label.pack(side='left', padx=5)

        # Menu déroulant pour la sélection de modèle
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, state='readonly')
        self.model_combobox['values'] = (
            'HoughLinesP (Segmented)',
            'HoughLinesP (Extended)',
            'Vanishing Lines',
            'RANSAC',
            'Edge Distance',
            'Intensity Profile',
            'Contour Hierarchy',
            'Random Forest Regressor',
            'MLP',
            'Gradient Boosting',
            'Support Vector Regressor'
        )
        self.model_combobox.current(0)  # Définir la sélection par défaut
        self.model_combobox.pack(side='left', padx=5)

        # Lier l'événement de sélection du menu déroulant à la méthode reset_image du contrôleur
        self.model_combobox.bind('<<ComboboxSelected>>', self.controller.reset_image)