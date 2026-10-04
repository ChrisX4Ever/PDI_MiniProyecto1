import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def segmentar_huella_ia(ruta_imagen):
    # 1. Cargar imagen en escala de grises
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"Error: No se encontró la imagen en '{ruta_imagen}'.")
        return None, None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplanar la imagen a un vector 1D de píxeles
    pixeles = img_gray.reshape((-1, 1))
    
    # 3. K-Means clustering (IA no supervisada) para encontrar 2 grupos (crestas vs fondo)
    print("Ejecutando segmentación con K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(pixeles)
    
    # 4. Reconstruir la máscara binaria
    mascara_ia = kmeans.labels_.reshape(img_gray.shape)
    
    # Asegurar que las crestas (más oscuras) queden en blanco (255) y el fondo en negro (0)
    centroides = kmeans.cluster_centers_.flatten()
    if centroides[0] > centroides[1]:
        mascara_ia = 1 - mascara_ia
        
    mascara_ia = (mascara_ia * 255).astype(np.uint8)
    
    return img_gray, mascara_ia

# ==========================================
# Ejecución directa y visualización
# ==========================================
# Cambia la ruta por la foto que quieras probar:
ruta_prueba = "Fotos_PDI_2026_Grupo1/d2_crema.jpg"

original, mascara = segmentar_huella_ia(ruta_prueba)

if original is not None and mascara is not None:
    # Crear ventana con Matplotlib para visualizar los resultados
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle('Segmentación por Inteligencia Artificial (K-Means)', fontsize=16)

    # Imagen original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Imagen Original (Grises)')
    axes[0].axis('off')

    # Máscara binaria predicha por la IA
    axes[1].imshow(mascara, cmap='gray')
    axes[1].set_title('Máscara Segmentada por IA')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()