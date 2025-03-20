import tkinter as tk
from tkinter import ttk

class Buttons:
    def __init__(self, parent, controller):
        self.parent = parent
        self.controller = controller

        self.button_frame = ttk.Frame(self.parent)
        self.button_frame.pack(fill='x', pady=10)

        # Ordre des boutons
        buttons = [
            ("Choisir un dossier (Ctrl+o)", self.controller.load_folder),
            ("Charger la vérité terrain (Ctrl+g)", self.controller.load_ground_truth),
            ("Charger depuis DB (Ctrl+d)", self.controller.load_from_db),
            ("Évaluer le set (Ctrl+e)", self.controller.evaluate_all_images)
        ]

        for text, command in buttons:
            btn = ttk.Button(
                self.button_frame,
                text=text,
                command=command
            )
            btn.pack(side='left', padx=5)