import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Funciones de Transformación ---

def aplicar_estiramiento(canal):
    """Aplica la transformación de estiramiento de contraste."""
    r_a = canal.min()
    r_b = canal.max()
    if r_a == r_b:
        return canal
    estirada = (canal.astype(np.float32) - r_a) * 255 / (r_b - r_a)
    return np.clip(estirada, 0, 255).astype(np.uint8)

def aplicar_ecualizacion(canal):
    """Aplica la ecualización de histograma global."""
    return cv2.equalizeHist(canal)

# --- Funciones de Graficado ---

def graficar_histograma(ax, canales, colores, labels, titulo):
    """Función para graficar histogramas con Matplotlib."""
    for canal, color, label in zip(canales, colores, labels):
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        ax.plot(hist, color=color, label=label)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlim([0, 256])
    if len(labels) > 1:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

def procesar_y_graficar_canal(axes_row, canal, cmap_orig, nombre_canal, transformacion_func, nombre_trans, color_hist):
    """
    Toma un solo canal, le aplica la transformación asignada y luego genera
    las dos estrategias de segmentación (Binaria 127 y Otsu), mostrando los resultados en 7 columnas.
    """
    # 1. Aplicar Transformación (Estiramiento o Ecualización)
    canal_trans = transformacion_func(canal)
    
    # 2. Aplicar Segmentaciones sobre el canal transformado
    _, binaria = cv2.threshold(canal_trans, 127, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(canal_trans, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Col 0: Canal Original
    axes_row[0].imshow(canal, cmap=cmap_orig)
    axes_row[0].set_title(f"Orig. {nombre_canal}", fontsize=11)
    axes_row[0].axis('off')
    
    # Col 1: Transformación (visualizado en gris para análisis de intensidad)
    axes_row[1].imshow(canal_trans, cmap='gray') 
    axes_row[1].set_title(f"{nombre_trans}", fontsize=11)
    axes_row[1].axis('off')
    
    # Col 2: Histograma Transformado
    graficar_histograma(axes_row[2], [canal_trans], [color_hist], [nombre_canal], f"Hist. {nombre_trans}")
    
    # Col 3: Segmentación Binaria
    axes_row[3].imshow(binaria, cmap='gray')
    axes_row[3].set_title("Seg. Binaria (127)", fontsize=11)
    axes_row[3].axis('off')
    
    # Col 4: Histograma Binaria (Típicamente dos líneas verticales en 0 y 255)
    graficar_histograma(axes_row[4], [binaria], ['black'], ['Bin'], "Hist. Binaria")
    
    # Col 5: Segmentación Otsu
    axes_row[5].imshow(otsu, cmap='gray')
    axes_row[5].set_title("Seg. Otsu", fontsize=11)
    axes_row[5].axis('off')
    
    # Col 6: Histograma Otsu
    graficar_histograma(axes_row[6], [otsu], ['black'], ['Otsu'], "Hist. Otsu")

# --- Lógica Principal de Visualización ---

def visualizar_analisis(grupo, modo):
    # Diccionario para mapear el número de grupo al prefijo base del archivo
    bases = {'1': 'd1', '2': 'd1_lat', '3': 'd2', '4': 'd2_lat'}
    base = bases[grupo]
    carpeta = "Fotos_PDI_2026_Grupo1"
    
    # ==========================================
    # CASO EXCEPCIÓN: RGB (Sin segmentación)
    # ==========================================
    if modo == 'RGB':
        # Se muestran las 3 condiciones originales para el grupo seleccionado
        rutas = [
            f"{carpeta}/{base}_seco.jpg",
            f"{carpeta}/{base}_agua.jpg",
            f"{carpeta}/{base}_crema.jpg"
        ]
        condiciones = ['Seco', 'Agua (Mojado)', 'Crema']
        
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
        fig.suptitle(f'Análisis Grupo {grupo} - Espacio: RGB (Fotos Originales)', fontsize=16)
        
        for i, (ruta, condicion) in enumerate(zip(rutas, condiciones)):
            img_bgr = cv2.imread(ruta)
            if img_bgr is None:
                print(f"Advertencia: No se pudo cargar '{ruta}'.")
                continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            r, g, b = cv2.split(img_rgb)
            
            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f'Original: {condicion}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(r, cmap='Reds')
            axes[i, 1].set_title('Canal R')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(g, cmap='Greens')
            axes[i, 2].set_title('Canal G')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(b, cmap='Blues')
            axes[i, 3].set_title('Canal B')
            axes[i, 3].axis('off')

            graficar_histograma(axes[i, 4], [r, g, b], ['red', 'green', 'blue'], ['R', 'G', 'B'], f'Hist. RGB ({condicion})')

        plt.tight_layout()
        plt.show()
        
    # ==========================================
    # CASOS CON TRANSFORMACIÓN Y SEGMENTACIÓN
    # ==========================================
    else:
        # Configurar la ruta y la transformación según la palabra clave ingresada
        if modo == 'HSV':
            condicion = 'seco'
            transformacion_func = aplicar_estiramiento
            nombre_trans = "Estirado"
            
        elif modo == 'CIELAB':
            condicion = 'crema'
            transformacion_func = aplicar_ecualizacion
            nombre_trans = "Ecualizado"
            
        elif modo == 'G':
            condicion = 'agua'
            transformacion_func = aplicar_estiramiento
            nombre_trans = "Estirado"
            
        ruta = f"{carpeta}/{base}_{condicion}.jpg"
        img_bgr = cv2.imread(ruta)
        
        if img_bgr is None:
            print(f"Error: No se pudo cargar la imagen '{ruta}'. Verifica que exista.")
            return
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Procesamiento para Escala de Grises (1 sola fila)
        if modo == 'G':
            fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(24, 4))
            fig.suptitle(f'Grupo {grupo} | Espacio: {modo} (Dedo {condicion}) | Transformación: {nombre_trans}', fontsize=16)
            
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            procesar_y_graficar_canal(axes, img_gray, 'gray', 'Gris', transformacion_func, nombre_trans, 'black')
            
        # Procesamiento para HSV y CIELAB (3 filas, una por canal)
        else:
            fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(26, 12))
            fig.suptitle(f'Grupo {grupo} | Espacio: {modo} (Dedo {condicion}) | Transformación: {nombre_trans}', fontsize=16)
            
            if modo == 'HSV':
                img_espacio = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                cmaps = ['hsv', 'gray', 'gray']
                nombres = ['H', 'S', 'V']
                colores_hist = ['purple', 'cyan', 'black']
            else: # CIELAB
                img_espacio = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                cmaps = ['gray', 'gray', 'gray']
                nombres = ['L', 'A', 'B']
                colores_hist = ['black', 'green', 'blue']
                
            canales = cv2.split(img_espacio)
            
            for i, canal in enumerate(canales):
                procesar_y_graficar_canal(axes[i], canal, cmaps[i], nombres[i], transformacion_func, nombre_trans, colores_hist[i])
                
        plt.tight_layout()
        plt.show()

# --- Interfaz de Usuario ---

grupos_validos = ['1', '2', '3', '4']
modos_validos = ['G', 'RGB', 'HSV', 'CIELAB']

while True:
    print("\n--- Visualizador Avanzado de Segmentación ---")
    print("Opciones de comando: [Grupo] [Modo]")
    print("- G      : Dedo con agua (Estiramiento + Segmentaciones)")
    print("- HSV    : Dedo seco (Estiramiento + Segmentaciones)")
    print("- CIELAB : Dedo con crema (Ecualización + Segmentaciones)")
    print("- RGB    : Originales de Seco, Agua y Crema (Sin segmentar)")
    print("Ejemplos: '1 CIELAB', '3 HSV', '2 G', '4 RGB'")
    
    entrada = input("Ingresa tu comando (o 'q' para salir): ").strip().upper()
    
    if entrada in ['Q', 'SALIR', 'EXIT']:
        print("Saliendo del visualizador.")
        break
        
    partes = entrada.split()
    
    if len(partes) != 2:
        print("Error: Debes ingresar el grupo y el modo separados por un espacio.")
        continue
        
    grupo, modo = partes[0], partes[1]
    
    if grupo not in grupos_validos:
        print(f"Error: El grupo '{grupo}' no es válido. Usa 1, 2, 3 o 4.")
        continue
        
    if modo not in modos_validos:
        print(f"Error: El modo '{modo}' no es válido. Usa G, RGB, HSV o CIELAB.")
        continue

    print(f"Generando análisis para el Grupo {grupo} en modo {modo}...")
    visualizar_analisis(grupo, modo)