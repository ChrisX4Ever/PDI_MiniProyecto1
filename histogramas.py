import cv2
import matplotlib.pyplot as plt

def analizar_imagen_especifica(ruta_imagen, num_imagen):
    # 1. Cargar imagen
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"Error: No se pudo cargar '{ruta_imagen}'. Verifica que exista.")
        return

    # Convertir a RGB base
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Extracción de los espacios requeridos
    # Escala de Grises tradicional
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Canal V del espacio HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, _, v = cv2.split(img_hsv)
    
    # Canal L del espacio CIELAB
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(img_lab)

    # 3. Configurar la figura 2x3
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    fig.suptitle(f'Análisis de Imagen {num_imagen} - Comparación de Espacios', fontsize=16)

    # --- FILA 1: Imágenes ---
    
    # Columna 0: Grises
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('Escala de Grises')
    axes[0, 0].axis('off')

    # Columna 1: Canal V (HSV)
    axes[0, 1].imshow(v, cmap='gray')
    axes[0, 1].set_title('Canal V (Brillo / HSV)')
    axes[0, 1].axis('off')

    # Columna 2: Canal L (CIELAB)
    axes[0, 2].imshow(l, cmap='gray')
    axes[0, 2].set_title('Canal L (Luminancia / CIELAB)')
    axes[0, 2].axis('off')

    # --- FILA 2: Histogramas ---
    
    # Lista de canales y configuraciones para iterar fácilmente
    canales_img = [img_gray, v, l]
    titulos_hist = ['Hist. Grises', 'Hist. Canal V', 'Hist. Canal L']
    colores_linea = ['black', 'blue', 'green'] # Colores para diferenciarlos visualmente

    for j in range(3):
        # Calcular histograma del canal correspondiente
        hist = cv2.calcHist([canales_img[j]], [0], None, [256], [0, 256])
        
        # Graficar
        axes[1, j].plot(hist, color=colores_linea[j])
        axes[1, j].set_xlim([0, 256])
        axes[1, j].set_title(titulos_hist[j])
        axes[1, j].grid(alpha=0.3)
        axes[1, j].set_xlabel('Intensidad (0-255)')
        axes[1, j].set_ylabel('Cantidad de Píxeles')

    plt.tight_layout()
    plt.show()

# --- Ejecución Principal e Interactiva ---

lista_imagenes = [
    "Fotos_PDI_2026_Grupo1/d1_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d1_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d2_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_crema.jpg", 
        "Fotos_PDI_2026_Grupo1/d2_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_crema.jpg"
]

while True:
    entrada = input("\nIngresa el número de imagen (1 al 12) o 'q' para salir: ").strip()
    
    if entrada.lower() in ['q', 'salir', 'exit']:
        print("Saliendo del visualizador.")
        break
        
    # Validar que la entrada sea un número dentro del rango permitido
    if entrada.isdigit() and 1 <= int(entrada) <= 12:
        indice = int(entrada) - 1
        ruta = lista_imagenes[indice]
        print(f"Procesando y generando visualización para la Imagen {entrada}...")
        analizar_imagen_especifica(ruta, entrada)
    else:
        print("Error: Ingresa un número válido entre 1 y 12.")