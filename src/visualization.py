# src/visualization.py
import cv2

def visualize_results(image, count):
    cv2.putText(image, f"Marches detectees: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Resultat", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()