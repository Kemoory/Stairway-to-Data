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

    def update_image_display(self, current_image, processed_image=None, debug_image=None, 
                            prediction=None, ground_truth=None):
        """
        Mettre à jour l'affichage avec les résultats.
        """
        if current_image is not None:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Clear canvas avant de redessiner
            self.canvas.delete("all")

            if processed_image is None or debug_image is None:
                # Conversion pour l'affichage unique
                img = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_width, img_height = img.size

                # Redimensionnement
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
                # Affichage côte à côte
                img_width = canvas_width // 2
                img_height = canvas_height

                # Image traitée
                processed_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                processed_img = Image.fromarray(processed_img)
                processed_img = processed_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.tk_processed_img = ImageTk.PhotoImage(processed_img)
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_processed_img)

                # Image de débogage
                debug_img = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                debug_img = Image.fromarray(debug_img)
                debug_img = debug_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.tk_debug_img = ImageTk.PhotoImage(debug_img)
                self.canvas.create_image(img_width, 0, anchor='nw', image=self.tk_debug_img)

            # Dessiner les résultats après l'image
            self._draw_results_overlay(prediction, ground_truth)

            # Mettre à jour le titre
            if self.controller.image_paths:
                filename = os.path.basename(self.controller.image_paths[self.controller.current_index])
                self.controller.title(f"Compteur de marches - {filename}")

    def _draw_results_overlay(self, prediction, ground_truth):
            """Dessiner l'overlay des résultats avec style."""
            # Position et dimensions
            box_width = 280
            box_height = 80 if ground_truth else 50
            x_offset = 20
            y_offset = 20
            
            # Style
            bg_color = '#2E3440'
            border_color = '#4C566A'
            text_color = '#ECEFF4'
            gt_color = '#A3BE8C'
            font = ('Helvetica', 12, 'bold')

            # Créer un cadre arrondi
            self.canvas.create_rectangle(
                x_offset, y_offset,
                x_offset + box_width, y_offset + box_height,
                fill=bg_color, outline=border_color, width=2
            )

            # Texte de prédiction
            self.canvas.create_text(
                x_offset + 15, y_offset + 15,
                text=f"Prédiction: {prediction}" if prediction else "Prédiction: N/A",
                fill=text_color,
                anchor='nw',
                font=font
            )

            # Texte de vérité terrain si disponible
            if ground_truth is not None:
                self.canvas.create_text(
                    x_offset + 15, y_offset + 45,
                    text=f"Vérité terrain: {ground_truth}",
                    fill=gt_color,
                    anchor='nw',
                    font=font
                )

                # Ajouter un indicateur de différence
                diff = abs(prediction - ground_truth)
                diff_text = f"±{diff}" if prediction else ""
                self.canvas.create_text(
                    x_offset + box_width - 20, y_offset + 30,
                    text=diff_text,
                    fill='#BF616A' if diff > 0 else gt_color,
                    anchor='ne',
                    font=('Helvetica', 14, 'bold')
                )