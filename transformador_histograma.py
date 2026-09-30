import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Funciones Auxiliares de Procesamiento ---

def aplicar_estiramiento(canal):
    """
    Aplica la transformación de estiramiento de contraste basada en el archivo image_5b8079.png.
    Utiliza fórmulas de numpy para ajustar el rango dinámico de un canal individual.
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
    Usa la función nativa de OpenCV cv2.equalizeHist para un canal individual.
    """
    return cv2.equalizeHist(canal)

# --- Función de Graficado ---

def graficar_histograma(ax, canales, colores, labels, titulo):
    """
    Función auxiliar para graficar histogramas superpuestos.
    Recibe un eje de Matplotlib, una lista de canales a analizar y sus colores/etiquetas.
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
    """
    Genera una cuadrícula de 3 filas (por imagen) y 6 columnas para análisis completo.
    Estructura de columnas: Original | Hist | Estirada (Visual) | Hist Estirada | Ecualizada (Visual) | Hist Ecualizada
    """
    # Se aumenta considerablemente el ancho de la figura para acomodar las 6 columnas
    ncols = 6
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(30, 15))
    fig.suptitle(f'Análisis Completo Grupo {grupo_num} - Espacio: {modo} (Visual y Estadístico)', fontsize=18)

    for i, ruta in enumerate(rutas_imagenes):
        # 1. Cargar imagen y convertir a RGB base
        img_bgr = cv2.imread(ruta)
        if img_bgr is None:
            print(f"Advertencia: No se pudo cargar '{ruta}'.")
            continue
            
        img_rgb_base = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Columna 0: Imagen Original
        axes[i, 0].imshow(img_rgb_base)
        axes[i, 0].set_title(f'Img {i + 1 + (int(grupo_num)-1)*3}: Original')
        axes[i, 0].axis('off')

        # 2. Lógica de transformaciones y visualización según el modo
        
        # Modo G: Escala de Grises
        if modo == 'G':
            img_original_base = cv2.cvtColor(img_rgb_base, cv2.COLOR_RGB2GRAY)
            
            # Aplicar transformaciones visuales (Grayscale)
            img_estirada_visual = aplicar_estiramiento(img_original_base)
            img_ecualizada_visual = aplicar_ecualizacion(img_original_base)
            
            colores_plot = ['black']
            labels_plot = ['Gris']
            
            # Graficar histogramas y las imágenes resultantes
            graficar_histograma(axes[i, 1], [img_original_base], colores_plot, labels_plot, 'Hist. Original')
            
            axes[i, 2].imshow(img_estirada_visual, cmap='gray')
            axes[i, 2].set_title('Grises Estirada')
            axes[i, 2].axis('off')
            graficar_histograma(axes[i, 3], [img_estirada_visual], colores_plot, labels_plot, 'Hist. Estirado')
            
            axes[i, 4].imshow(img_ecualizada_visual, cmap='gray')
            axes[i, 4].set_title('Grises Ecualizada')
            axes[i, 4].axis('off')
            graficar_histograma(axes[i, 5], [img_ecualizada_visual], colores_plot, labels_plot, 'Hist. Ecualizada')

        # Modos Multicanal
        else:
            # Seleccionar el espacio de color de trabajo y sus configuraciones
            conv_base = None
            conv_back = None
            if modo == 'RGB':
                # No se requiere conversión inicial, ya es RGB
                img_original_espacio = img_rgb_base
                conv_back = None
                colores_plot = ['red', 'green', 'blue']
                labels_plot = ['R', 'G', 'B']
            elif modo == 'HSV':
                img_original_espacio = cv2.cvtColor(img_rgb_base, cv2.COLOR_RGB2HSV)
                conv_back = cv2.COLOR_HSV2RGB
                colores_plot = ['purple', 'cyan', 'black']
                labels_plot = ['H', 'S', 'V']
            elif modo == 'CIELAB':
                img_original_espacio = cv2.cvtColor(img_rgb_base, cv2.COLOR_RGB2LAB)
                conv_back = cv2.COLOR_LAB2RGB
                colores_plot = ['black', 'green', 'blue']
                labels_plot = ['L', 'A', 'B']

            # Separar los 3 canales
            c1, c2, c3 = cv2.split(img_original_espacio)
            
            # Aplicar transformaciones por canal individualmente
            c1_est = aplicar_estiramiento(c1)
            c2_est = aplicar_estiramiento(c2)
            c3_est = aplicar_estiramiento(c3)
            
            c1_eq = aplicar_ecualizacion(c1)
            c2_eq = aplicar_ecualizacion(c2)
            c3_eq = aplicar_ecualizacion(c3)
            
            # Combinar canales y reconvertir a RGB para visualización
            img_espacio_est = cv2.merge([c1_est, c2_est, c3_est])
            img_espacio_eq = cv2.merge([c1_eq, c2_eq, c3_eq])
            
            img_rgb_est_visual = cv2.cvtColor(img_espacio_est, conv_back) if conv_back else img_espacio_est
            img_rgb_eq_visual = cv2.cvtColor(img_espacio_eq, conv_back) if conv_back else img_espacio_eq
            
            # Graficar histogramas y las imágenes resultantes (reconversión a RGB)
            graficar_histograma(axes[i, 1], [c1, c2, c3], colores_plot, labels_plot, f'Hist. Original ({modo})')
            
            axes[i, 2].imshow(img_rgb_est_visual)
            axes[i, 2].set_title(f'{modo} Estirada (Visual)')
            axes[i, 2].axis('off')
            graficar_histograma(axes[i, 3], [c1_est, c2_est, c3_est], colores_plot, labels_plot, f'Hist. Estirado ({modo})')
            
            axes[i, 4].imshow(img_rgb_eq_visual)
            axes[i, 4].set_title(f'{modo} Ecualizada (Visual)')
            axes[i, 4].axis('off')
            graficar_histograma(axes[i, 5], [c1_eq, c2_eq, c3_eq], colores_plot, labels_plot, f'Hist. Ecualizada ({modo})')

    plt.tight_layout()
    plt.show()

# --- Ejecución Principal e Interactiva ---

# Lista de rutas (ejemplo de estructura de carpetas)
lista_imagenes = [
    "Fotos_PDI_2026_Grupo1/d1_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d1_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d2_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d2_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_crema.jpg"
]

modos_validos = ['G', 'RGB', 'HSV', 'CIELAB']
grupos_validos = ['1', '2', '3', '4']

while True:
    print("\n--- Visualizador Completo ---")
    print("Opciones de comando: [Grupo] [Modo]")
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

    # Calcular índices para seleccionar el grupo de 3 imágenes
    indice = int(grupo) - 1
    inicio = indice * 3
    fin = inicio + 3
    grupo_actual = lista_imagenes[inicio:fin]
    
    print(f"Generando visualización completa {modo} para el Grupo {grupo}...")
    visualizar_grupo(grupo_actual, grupo, modo)
