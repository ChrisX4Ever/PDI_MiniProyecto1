import cv2
import matplotlib.pyplot as plt

def visualizar_grupo(rutas_imagenes, grupo_num, modo):
    # Definir la cantidad de columnas según el modo seleccionado
    ncols = 3 if modo == 'G' else 4
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(16, 9))
    fig.suptitle(f'Análisis Grupo {grupo_num} - Espacio: {modo}', fontsize=16)

    for i, ruta in enumerate(rutas_imagenes):
        # 1. Cargar imagen y convertir a RGB base
        img_bgr = cv2.imread(ruta)
        if img_bgr is None:
            print(f"Advertencia: No se pudo cargar '{ruta}'.")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Columna 0: Imagen Original (común para todos los modos)
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f'Img {i + 1 + (int(grupo_num)-1)*3}: Original')
        axes[i, 0].axis('off')

        # 2. Lógica de visualización según el modo
        if modo == 'G':
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            axes[i, 1].imshow(img_gray, cmap='gray')
            axes[i, 1].set_title('Escala de Grises')
            axes[i, 1].axis('off')
            
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            axes[i, 2].plot(hist, color='black')
            axes[i, 2].set_xlim([0, 256])
            axes[i, 2].set_title('Histograma')
            axes[i, 2].grid(alpha=0.3)

        elif modo == 'RGB':
            r, g, b = cv2.split(img_rgb)
            
            axes[i, 1].imshow(r, cmap='Reds')
            axes[i, 1].set_title('Canal R')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(g, cmap='Greens')
            axes[i, 2].set_title('Canal G')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(b, cmap='Blues')
            axes[i, 3].set_title('Canal B')
            axes[i, 3].axis('off')

        elif modo == 'HSV':
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(img_hsv)
            
            axes[i, 1].imshow(h, cmap='hsv')
            axes[i, 1].set_title('Canal H (Matiz)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(s, cmap='gray')
            axes[i, 2].set_title('Canal S (Saturación)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(v, cmap='gray')
            axes[i, 3].set_title('Canal V (Brillo)')
            axes[i, 3].axis('off')

        elif modo == 'CIELAB':
            # Conversión a CIELAB
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b_chan = cv2.split(img_lab)
            
            axes[i, 1].imshow(l, cmap='gray')
            axes[i, 1].set_title('Canal L (Luminancia)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(a, cmap='gray')
            axes[i, 2].set_title('Canal A (Verde-Rojo)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(b_chan, cmap='gray')
            axes[i, 3].set_title('Canal B (Azul-Amarillo)')
            axes[i, 3].axis('off')

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
    
    # Validar que se ingresaron exactamente dos parámetros
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