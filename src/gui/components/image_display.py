import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os

class ImageDisplay:
    def __init__(self, parent, controller):
        """
        Initialiser le composant d'affichage d'image.

        Args:
            parent: Le widget parent (par exemple, un cadre).
            controller: Le contrôleur principal de l'application (instance de Interface).
        """
        self.parent = parent
        self.controller = controller  # Référence à la classe principale Interface

        # Créer le canvas pour afficher les images
        self.canvas = tk.Canvas(self.parent, bg='#3B4252', highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')

        # Label pour les informations de statut
        self.info_label = ttk.Label(self.parent, text="[← →] Naviguer | [T] Traiter", style='Status.TLabel')
        self.info_label.pack(pady=10)

    def update_image_display(self, current_image, processed_image=None, debug_image=None):
        """
        Mettre à jour l'affichage de l'image sur le canvas.

        Args:
            current_image: L'image actuelle à afficher.
            processed_image: L'image traitée (optionnel).
            debug_image: L'image de débogage (optionnel).
        """
        if current_image is not None:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            self.canvas.delete("all")

            if processed_image is None or debug_image is None:
                img = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_width, img_height = img.size

                img_aspect_ratio = img_width / img_height
                canvas_aspect_ratio = canvas_width / canvas_height

                if img_aspect_ratio > canvas_aspect_ratio:
                    new_width = canvas_width
                    new_height = int(canvas_width / img_aspect_ratio)
                else:
                    new_height = canvas_height
                    new_width = int(canvas_height * img_aspect_ratio)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2

                self.tk_img = ImageTk.PhotoImage(img)
                self.canvas.create_image(x_offset, y_offset, anchor='nw', image=self.tk_img)
            else:
                img_width = canvas_width // 2
                img_height = canvas_height

                processed_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                processed_img = Image.fromarray(processed_img)
                processed_img = processed_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.tk_processed_img = ImageTk.PhotoImage(processed_img)
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_processed_img)

                debug_img = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                debug_img = Image.fromarray(debug_img)
                debug_img = debug_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.tk_debug_img = ImageTk.PhotoImage(debug_img)
                self.canvas.create_image(img_width, 0, anchor='nw', image=self.tk_debug_img)

            self.controller.title(f"Compteur de marches - {os.path.basename(self.controller.image_paths[self.controller.current_index])}")