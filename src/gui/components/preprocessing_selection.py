import tkinter as tk
from tkinter import ttk

class PreprocessingSelection:
    def __init__(self, parent, controller):
        """
        Initialiser le composant de prétraitement.

        Args:
            parent: Le widget parent (par exemple, un cadre).
            controller: Le contrôleur principal de l'application (instance de Interface).
        """
        self.parent = parent
        self.controller = controller  # Référence à la classe principale Interface

        # Créer le cadre de prétraitement
        self.preprocess_frame = ttk.Frame(self.parent)
        self.preprocess_frame.pack(fill='x', pady=5)

        # Étiquette pour le menu déroulant de prétraitement
        self.preprocess_label = ttk.Label(self.preprocess_frame, text="Choix du prétraitement :")
        self.preprocess_label.pack(side='left', padx=5)

        # Menu déroulant pour la sélection du prétraitement
        self.preprocess_var = tk.StringVar()
        self.preprocess_combobox = ttk.Combobox(self.preprocess_frame, textvariable=self.preprocess_var, state='readonly')
        self.preprocess_combobox['values'] = (
            '(None)',
            'Gaussian Blur + Canny',
            'Median Blur + Canny',
            'Split and Merge',
            'Adaptive Thresholding',
            'Gradient Orientation',
            'Homomorphic Filter',
            'Phase Congruency',
            'Wavelet Transform',
        )
        self.preprocess_combobox.current(0)  # Définir la sélection par défaut
        self.preprocess_combobox.pack(side='left', padx=5)

        # Lier l'événement de sélection du menu déroulant à la méthode reset_image du contrôleur
        self.preprocess_combobox.bind('<<ComboboxSelected>>', self.controller.reset_image)