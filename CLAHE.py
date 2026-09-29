import cv2
import matplotlib.pyplot as plt

def aplicar_clahe(ruta_imagen, num_imagen, escala):
    # Cargar y convertir imagen base a RGB
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"Error: No se pudo cargar '{ruta_imagen}'. Verifica que exista.")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Inicializar el objeto CLAHE
    # clipLimit define el umbral de contraste; tileGridSize divide la imagen en bloques (8x8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if escala == 'L':
        # Convertir a CIELAB y separar canales
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        canal_orig, a, b = cv2.split(img_lab)
        
        # Aplicar CLAHE solo al canal L (Luminancia)
        canal_clahe = clahe.apply(canal_orig)
        
        # Reconstruir la imagen de vuelta a RGB
        lab_clahe = cv2.merge((canal_clahe, a, b))
        img_reconstruida = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        titulo_canal = 'Canal L (CIELAB)'
        titulo_reconstruccion = 'RGB Reconstruido'
        cmap_reconstruccion = None
        
    elif escala == 'G':
        # Convertir directamente a escala de grises
        canal_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Aplicar CLAHE
        canal_clahe = clahe.apply(canal_orig)
        
        # Al no haber canales de color, la "reconstrucción" es simplemente la imagen gris resultante
        img_reconstruida = canal_clahe
        
        titulo_canal = 'Escala de Grises'
        titulo_reconstruccion = 'Grises Post CLAHE'
        cmap_reconstruccion = 'gray'

    # Configurar la figura 2x3
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
    fig.suptitle(f'Aplicación de CLAHE - Imagen {num_imagen} - Escala: {escala}', fontsize=16)

    # --- FILA 1: ESTADO ORIGINAL ---
    
    # Columna 1: Canal Original
    axes[0, 0].imshow(canal_orig, cmap='gray')
    axes[0, 0].set_title(f'Original: {titulo_canal}')
    axes[0, 0].axis('off')
    
    # Columna 2: Histograma del Canal Original
    hist_orig = cv2.calcHist([canal_orig], [0], None, [256], [0, 256])
    axes[0, 1].plot(hist_orig, color='black')
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].set_title('Histograma Original')
    axes[0, 1].grid(alpha=0.3)
    
    # Columna 3: Imagen RGB Original
    axes[0, 2].imshow(img_rgb)
    axes[0, 2].set_title('RGB Original')
    axes[0, 2].axis('off')

    # --- FILA 2: POST CLAHE ---
    
    # Columna 1: Canal Post CLAHE
    axes[1, 0].imshow(canal_clahe, cmap='gray')
    axes[1, 0].set_title(f'CLAHE: {titulo_canal}')
    axes[1, 0].axis('off')
    
    # Columna 2: Histograma del Canal Post CLAHE
    hist_clahe = cv2.calcHist([canal_clahe], [0], None, [256], [0, 256])
    axes[1, 1].plot(hist_clahe, color='blue')
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].set_title('Histograma Post CLAHE')
    axes[1, 1].grid(alpha=0.3)
    
    # Columna 3: Imagen Reconstruida / Final
    axes[1, 2].imshow(img_reconstruida, cmap=cmap_reconstruccion)
    axes[1, 2].set_title(titulo_reconstruccion)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# --- Ejecución Principal e Interactiva ---

lista_imagenes = [
    "Fotos_PDI_2026_Grupo1/d1_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d1_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d2_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_crema.jpg", 
        "Fotos_PDI_2026_Grupo1/d2_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_crema.jpg"
]

escalas_validas = ['G', 'L']

while True:
    print("\nFormato de entrada: [Número de Imagen] [Escala]")
    print("Ejemplos: '1 L' (Imagen 1 en Luminancia), '12 G' (Imagen 12 en Grises)")
    entrada = input("Ingresa tu comando (o 'q' para salir): ").strip().upper()
    
    if entrada in ['Q', 'SALIR', 'EXIT']:
        print("Saliendo del visualizador.")
        break
        
    partes = entrada.split()
    
    if len(partes) != 2:
        print("Error: Debes ingresar el número de imagen y la escala separados por un espacio.")
        continue
        
    num_str, escala = partes[0], partes[1]
    
    if not (num_str.isdigit() and 1 <= int(num_str) <= 12):
        print(f"Error: El número '{num_str}' no es válido. Usa un valor del 1 al 12.")
        continue
        
    if escala not in escalas_validas:
        print(f"Error: La escala '{escala}' no es válida. Usa G o L.")
        continue

    # Calcular índice de la lista (0 a 11)
    indice = int(num_str) - 1
    ruta = lista_imagenes[indice]
    
    print(f"Aplicando CLAHE a la Imagen {num_str} en el espacio {escala}...")
    aplicar_clahe(ruta, num_str, escala)