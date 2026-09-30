import cv2
import matplotlib.pyplot as plt
import numpy as np

def aplicar_estiramiento(canal):
    """
    Aplica la transformación de estiramiento de contraste basada en el archivo image_5b8079.png.
    """
    r_a = canal.min()
    r_b = canal.max()
    
    # Evitar división por cero si la imagen es de un color completamente uniforme
    if r_a == r_b:
        return canal
        
    estirada = (canal.astype(np.float32) - r_a) * 255 / (r_b - r_a)
    estirada = np.clip(estirada, 0, 255).astype(np.uint8)
    return estirada

def aplicar_ecualizacion(canal):
    """
    Aplica la ecualización de histograma global basada en el archivo image_5bda2f.png.
    """
    return cv2.equalizeHist(canal)

def graficar_histograma(ax, canales, colores, labels, titulo):
    """
    Función auxiliar para graficar histogramas superpuestos.
    """
    for canal, color, label in zip(canales, colores, labels):
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        ax.plot(hist, color=color, label=label)
        
    ax.set_title(titulo)
    ax.set_xlim([0, 256])
    if len(labels) > 1:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

def visualizar_grupo(rutas_imagenes, grupo_num, modo):
    # 4 columnas: Imagen Original | Hist. Original | Hist. Estiramiento | Hist. Ecualización
    ncols = 4
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(20, 12))
    fig.suptitle(f'Análisis Grupo {grupo_num} - Espacio: {modo} (Comparación de Transformaciones)', fontsize=16)

    for i, ruta in enumerate(rutas_imagenes):
        # 1. Cargar imagen y convertir a RGB base
        img_bgr = cv2.imread(ruta)
        if img_bgr is None:
            print(f"Advertencia: No se pudo cargar '{ruta}'.")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Columna 0: Imagen Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f'Img {i + 1 + (int(grupo_num)-1)*3}: Original')
        axes[i, 0].axis('off')

        # 2. Lógica de transformaciones y visualización según el modo
        if modo == 'G':
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # Aplicar transformaciones
            img_estirada = aplicar_estiramiento(img_gray)
            img_ecualizada = aplicar_ecualizacion(img_gray)
            
            # Graficar histogramas
            graficar_histograma(axes[i, 1], [img_gray], ['black'], ['Gris'], 'Hist. Original')
            graficar_histograma(axes[i, 2], [img_estirada], ['black'], ['Gris'], 'Hist. Estiramiento')
            graficar_histograma(axes[i, 3], [img_ecualizada], ['black'], ['Gris'], 'Hist. Ecualización Global')

        else:
            # Seleccionar el espacio de color y sus configuraciones
            if modo == 'RGB':
                img_espacio = img_rgb
                colores = ['red', 'green', 'blue']
                labels = ['R', 'G', 'B']
            elif modo == 'HSV':
                img_espacio = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                colores = ['purple', 'cyan', 'black']
                labels = ['H', 'S', 'V']
            elif modo == 'CIELAB':
                img_espacio = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                colores = ['black', 'green', 'blue']
                labels = ['L', 'A', 'B']

            # Separar los 3 canales
            c1, c2, c3 = cv2.split(img_espacio)
            
            # Aplicar transformaciones por canal individualmente
            c1_est = aplicar_estiramiento(c1)
            c2_est = aplicar_estiramiento(c2)
            c3_est = aplicar_estiramiento(c3)
            
            c1_eq = aplicar_ecualizacion(c1)
            c2_eq = aplicar_ecualizacion(c2)
            c3_eq = aplicar_ecualizacion(c3)
            
            # Graficar histogramas superpuestos
            graficar_histograma(axes[i, 1], [c1, c2, c3], colores, labels, f'Hist. Original ({modo})')
            graficar_histograma(axes[i, 2], [c1_est, c2_est, c3_est], colores, labels, f'Hist. Estirado ({modo})')
            graficar_histograma(axes[i, 3], [c1_eq, c2_eq, c3_eq], colores, labels, f'Hist. Ecualizado ({modo})')

    plt.tight_layout()
    plt.show()

# --- Ejecución Principal e Interactiva ---

lista_imagenes = [
    "Fotos_PDI_2026_Grupo1/d1_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d1_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d2_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d2_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_crema.jpg"
]

modos_validos = ['G', 'RGB', 'HSV', 'CIELAB']
grupos_validos = ['1', '2', '3', '4']

while True:
    print("\nOpciones de comando: [Grupo] [Modo]")
    print("Ejemplos: '1 CIELAB', '3 HSV', '2 G', '4 RGB'")
    entrada = input("Ingresa tu comando (o 'q' para salir): ").strip().upper()
    
    if entrada in ['Q', 'SALIR', 'EXIT']:
        print("Saliendo del visualizador.")
        break
        
    partes = entrada.split()
    
    if len(partes) != 2:
        print("Error: Debes ingresar el grupo y el canal separados por un espacio.")
        continue
        
    grupo, modo = partes[0], partes[1]
    
    if grupo not in grupos_validos:
        print(f"Error: El grupo '{grupo}' no es válido. Usa 1, 2, 3 o 4.")
        continue
        
    if modo not in modos_validos:
        print(f"Error: El canal '{modo}' no es válido. Usa G, RGB, HSV o CIELAB.")
        continue

    # Calcular índices
    indice = int(grupo) - 1
    inicio = indice * 3
    fin = inicio + 3
    grupo_actual = lista_imagenes[inicio:fin]
    
    print(f"Generando visualización {modo} para el Grupo {grupo}...")
    visualizar_grupo(grupo_actual, grupo, modo)
