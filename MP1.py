import cv2
import matplotlib.pyplot as plt

def mostrar_grupo(rutas_imagenes, grupo_num):
    # Crear una única figura de 3 filas y 6 columnas
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 9))
    fig.suptitle(f'Análisis de Imágenes - Grupo {grupo_num}', fontsize=16)

    for i, ruta in enumerate(rutas_imagenes):
        # 1. Cargar y convertir colores
        img_bgr = cv2.imread(ruta)
        
        if img_bgr is None:
            print(f"Advertencia: No se pudo cargar '{ruta}'.")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        r, g, b = cv2.split(img_rgb)

        # 2. Visualización de imágenes (Columnas 0 a 4)
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f'Img {i + 1 + (int(grupo_num)-1)*3}: Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_gray, cmap='gray')
        axes[i, 1].set_title('Escala de Grises')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(r, cmap='Reds')
        axes[i, 2].set_title('Canal Rojo')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(g, cmap='Greens')
        axes[i, 3].set_title('Canal Verde')
        axes[i, 3].axis('off')

        axes[i, 4].imshow(b, cmap='Blues')
        axes[i, 4].set_title('Canal Azul')
        axes[i, 4].axis('off')

        # 3. Cálculo y visualización del Histograma en Grises (Columna 5)
        hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        axes[i, 5].plot(hist_gray, color='black')
        axes[i, 5].set_xlim([0, 256])
        axes[i, 5].set_title('Histograma (Gris)')
        axes[i, 5].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- Ejecución Principal e Interactiva ---

lista_imagenes = [
    "Fotos_PDI_2026_Grupo1/d1_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d1_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d1_lat_crema.jpg", 
    "Fotos_PDI_2026_Grupo1/d2_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_crema.jpg", 
        "Fotos_PDI_2026_Grupo1/d2_lat_seco.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_agua.jpg", "Fotos_PDI_2026_Grupo1/d2_lat_crema.jpg"
]

# Bucle infinito para recibir comandos desde la terminal
while True:
    comando = input("\nIngresa un número (1 al 4) para ver un grupo, o 'q' para salir: ")
    
    if comando.lower() in ['q', 'salir', 'exit']:
        print("Saliendo del visualizador.")
        break
        
    if comando in ['1', '2', '3', '4']:
        indice = int(comando) - 1
        inicio = indice * 3
        fin = inicio + 3
        
        grupo_actual = lista_imagenes[inicio:fin]
        print(f"Generando gráfico para el Grupo {comando} (Imágenes {inicio+1} a {fin})...")
        
        # El programa pausará el bucle aquí hasta que cierres la ventana de Matplotlib
        mostrar_grupo(grupo_actual, comando)
    else:
        print("Comando no reconocido. Por favor ingresa 1, 2, 3, 4 o 'q'.")