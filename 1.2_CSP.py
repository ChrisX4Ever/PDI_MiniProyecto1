import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Definir nombres de imágenes
imgNames = ['d1_agua.jpg', 'd1_crema.jpg', 'd1_lat_agua.jpg', 'd1_lat_crema.jpg', 'd1_lat_seco.jpg', 'd1_seco.jpg', 'd2_agua.jpg', 'd2_crema.jpg', 'd2_lat_agua.jpg', 'd2_lat_crema.jpg', 'd2_lat_seco.jpg', 'd2_seco.jpg']

directorio_script = os.path.dirname(os.path.abspath(__file__))
carpeta_imagenes = os.path.join(directorio_script, 'Fotos_PDI_2026_Grupo1')

# 2. Crear ventana con dimensiones ampliadas
fig, axes = plt.subplots(nrows=12, ncols=6, figsize=(24, 35))
fig.suptitle("Análisis completo de imágenes de huellas", fontsize=22, y=0.98, fontweight='bold')

titulos_columnas = ['Original', 'Grises', 'Canal R (Rojo)', 'Canal G (Verde)', 'Canal B (Azul)', 'Histograma Original']

# 3. Iterar sobre todas las imágenes
for i, nombre_imagen in enumerate(imgNames):
    ruta = os.path.join(carpeta_imagenes, nombre_imagen)
    
    img_bgr = cv.imread(ruta)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    roi_rgb = img_rgb[100:400, 150:450]
    gray = cv.cvtColor(roi_rgb, cv.COLOR_RGB2GRAY)
    
    # Crear canales coloreados puros
    roi_R = np.zeros_like(roi_rgb)
    roi_G = np.zeros_like(roi_rgb)
    roi_B = np.zeros_like(roi_rgb)
    
    roi_R[:,:,0] = roi_rgb[:,:,0] 
    roi_G[:,:,1] = roi_rgb[:,:,1] 
    roi_B[:,:,2] = roi_rgb[:,:,2] 
    
    # --- GRAFICAR ---
    # Imagen Original
    axes[i, 0].imshow(roi_rgb)
    axes[i, 0].axis('off')
    # Colocar el nombre de la imagen a la izquierda para ahorrar espacio vertical
    axes[i, 0].text(-0.1, 0.5, nombre_imagen, transform=axes[i, 0].transAxes, 
                    fontsize=14, fontweight='bold', ha='right', va='center')
    
    # Escala de grises y Canales
    axes[i, 1].imshow(gray, cmap='gray')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(roi_R)
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(roi_G)
    axes[i, 3].axis('off')
    
    axes[i, 4].imshow(roi_B)
    axes[i, 4].axis('off')
    
    # Títulos de columna solo en la primera fila
    if i == 0:
        for j in range(6):
            axes[i, j].set_title(titulos_columnas[j], fontsize=15, pad=15)
    
    # Histograma de los 3 canales
    for canal, color, nombre in [(0, "red", "R"), (1, "green", "G"), (2, "blue", "B")]:
        hist = cv.calcHist([roi_rgb], [canal], None, [256], [0, 256]).ravel()
        axes[i, 5].plot(hist, color=color, alpha=0.8, label=nombre, linewidth=2)
        
    if i == 0: 
        axes[i, 5].legend(fontsize='11', loc='upper right')
        
    axes[i, 5].set_xlim([0, 256])
    axes[i, 5].set_yticks([]) # Ocultar eje Y
    axes[i, 5].grid(alpha=0.3)

# 4. Forzar la eliminación de espacios en blanco (wspace=horizontal, hspace=vertical)
plt.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.02, wspace=0.05, hspace=0.1)

# 5. Exportar la imagen en altísima resolución antes de mostrarla en pantalla
# Esto guardará un archivo llamado "analisis_huellas.png" en tu carpeta
plt.savefig('analisis_huellas.png', dpi=300, bbox_inches='tight')

# Mostrar la ventana en vivo
plt.show()


# 1. Definir únicamente las 4 imágenes solicitadas
imgNames_seco = [
    'd1_seco.jpg', 
    'd1_lat_seco.jpg', 
    'd2_seco.jpg', 
    'd2_lat_seco.jpg'
]

imgNames_agua = [
    'd1_agua.jpg', 
    'd1_lat_agua.jpg', 
    'd2_agua.jpg', 
    'd2_lat_agua.jpg'
]

imgNames_crema = [
    'd1_crema.jpg', 
    'd1_lat_crema.jpg', 
    'd2_crema.jpg', 
    'd2_lat_crema.jpg'
]

directorio_script = os.path.dirname(os.path.abspath(__file__))
carpeta_imagenes = os.path.join(directorio_script, 'Fotos_PDI_2026_Grupo1')

# 2. Crear ventana con 4 filas y 6 columnas (tamaño ajustado para 4 imágenes)
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(22, 12))
fig.suptitle("Análisis detallado - Dedo Seco", fontsize=20, y=0.96, fontweight='bold')

titulos_columnas = ['Original', 'Grises', 'Canal R (Rojo)', 'Canal G (Verde)', 'Canal B (Azul)', 'Histograma Original']

# 3. Iterar solo sobre las 4 imágenes
for i, nombre_imagen in enumerate(imgNames_seco):
    ruta = os.path.join(carpeta_imagenes, nombre_imagen)
    
    img_bgr = cv.imread(ruta)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    roi_rgb = img_rgb[100:400, 150:450]
    gray = cv.cvtColor(roi_rgb, cv.COLOR_RGB2GRAY)
    
    # Crear canales coloreados puros
    roi_R = np.zeros_like(roi_rgb)
    roi_G = np.zeros_like(roi_rgb)
    roi_B = np.zeros_like(roi_rgb)
    
    roi_R[:,:,0] = roi_rgb[:,:,0] 
    roi_G[:,:,1] = roi_rgb[:,:,1] 
    roi_B[:,:,2] = roi_rgb[:,:,2] 
    
    # --- GRAFICAR ---
    # Imagen Original con nombre a la izquierda
    axes[i, 0].imshow(roi_rgb)
    axes[i, 0].axis('off')
    axes[i, 0].text(-0.1, 0.5, nombre_imagen, transform=axes[i, 0].transAxes, 
                    fontsize=16, fontweight='bold', ha='right', va='center')
    
    # Escala de grises y Canales
    axes[i, 1].imshow(gray, cmap='gray')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(roi_R)
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(roi_G)
    axes[i, 3].axis('off')
    
    axes[i, 4].imshow(roi_B)
    axes[i, 4].axis('off')
    
    # Títulos de columna solo en la primera fila
    if i == 0:
        for j in range(6):
            axes[i, j].set_title(titulos_columnas[j], fontsize=16, pad=15)
    
    # Histograma de los 3 canales
    for canal, color, nombre in [(0, "red", "R"), (1, "green", "G"), (2, "blue", "B")]:
        hist = cv.calcHist([roi_rgb], [canal], None, [256], [0, 256]).ravel()
        axes[i, 5].plot(hist, color=color, alpha=0.8, label=nombre, linewidth=2)
        
    if i == 0: 
        axes[i, 5].legend(fontsize='12', loc='upper right')
        
    axes[i, 5].set_xlim([0, 256])
    axes[i, 5].set_yticks([]) # Ocultar eje Y
    axes[i, 5].grid(alpha=0.3)

# 4. Ajustar márgenes para maximizar el tamaño de las imágenes
plt.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.05, wspace=0.05, hspace=0.1)

# Guardar y mostrar
plt.savefig('seleccion_cuatro_huellas_seco.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Crear ventana con 4 filas y 6 columnas (tamaño ajustado para 4 imágenes)
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(22, 12))
fig.suptitle("Análisis detallado - Dedo Agua", fontsize=20, y=0.96, fontweight='bold')

titulos_columnas = ['Original', 'Grises', 'Canal R (Rojo)', 'Canal G (Verde)', 'Canal B (Azul)', 'Histograma Original']

# 3. Iterar solo sobre las 4 imágenes
for i, nombre_imagen in enumerate(imgNames_agua):
    ruta = os.path.join(carpeta_imagenes, nombre_imagen)
    
    img_bgr = cv.imread(ruta)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    roi_rgb = img_rgb[100:400, 150:450]
    gray = cv.cvtColor(roi_rgb, cv.COLOR_RGB2GRAY)
    
    # Crear canales coloreados puros
    roi_R = np.zeros_like(roi_rgb)
    roi_G = np.zeros_like(roi_rgb)
    roi_B = np.zeros_like(roi_rgb)
    
    roi_R[:,:,0] = roi_rgb[:,:,0] 
    roi_G[:,:,1] = roi_rgb[:,:,1] 
    roi_B[:,:,2] = roi_rgb[:,:,2] 
    
    # --- GRAFICAR ---
    # Imagen Original con nombre a la izquierda
    axes[i, 0].imshow(roi_rgb)
    axes[i, 0].axis('off')
    axes[i, 0].text(-0.1, 0.5, nombre_imagen, transform=axes[i, 0].transAxes, 
                    fontsize=16, fontweight='bold', ha='right', va='center')
    
    # Escala de grises y Canales
    axes[i, 1].imshow(gray, cmap='gray')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(roi_R)
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(roi_G)
    axes[i, 3].axis('off')
    
    axes[i, 4].imshow(roi_B)
    axes[i, 4].axis('off')
    
    # Títulos de columna solo en la primera fila
    if i == 0:
        for j in range(6):
            axes[i, j].set_title(titulos_columnas[j], fontsize=16, pad=15)
    
    # Histograma de los 3 canales
    for canal, color, nombre in [(0, "red", "R"), (1, "green", "G"), (2, "blue", "B")]:
        hist = cv.calcHist([roi_rgb], [canal], None, [256], [0, 256]).ravel()
        axes[i, 5].plot(hist, color=color, alpha=0.8, label=nombre, linewidth=2)
        
    if i == 0: 
        axes[i, 5].legend(fontsize='12', loc='upper right')
        
    axes[i, 5].set_xlim([0, 256])
    axes[i, 5].set_yticks([]) # Ocultar eje Y
    axes[i, 5].grid(alpha=0.3)

# 4. Ajustar márgenes para maximizar el tamaño de las imágenes
plt.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.05, wspace=0.05, hspace=0.1)

# Guardar y mostrar
plt.savefig('seleccion_cuatro_huellas_agua.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Crear ventana con 4 filas y 6 columnas (tamaño ajustado para 4 imágenes)
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(22, 12))
fig.suptitle("Análisis detallado - Dedo Crema", fontsize=20, y=0.96, fontweight='bold')

titulos_columnas = ['Original', 'Grises', 'Canal R (Rojo)', 'Canal G (Verde)', 'Canal B (Azul)', 'Histograma Original']

# 3. Iterar solo sobre las 4 imágenes
for i, nombre_imagen in enumerate(imgNames_crema):
    ruta = os.path.join(carpeta_imagenes, nombre_imagen)
    
    img_bgr = cv.imread(ruta)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    roi_rgb = img_rgb[100:400, 150:450]
    gray = cv.cvtColor(roi_rgb, cv.COLOR_RGB2GRAY)
    
    # Crear canales coloreados puros
    roi_R = np.zeros_like(roi_rgb)
    roi_G = np.zeros_like(roi_rgb)
    roi_B = np.zeros_like(roi_rgb)
    
    roi_R[:,:,0] = roi_rgb[:,:,0] 
    roi_G[:,:,1] = roi_rgb[:,:,1] 
    roi_B[:,:,2] = roi_rgb[:,:,2] 
    
    # --- GRAFICAR ---
    # Imagen Original con nombre a la izquierda
    axes[i, 0].imshow(roi_rgb)
    axes[i, 0].axis('off')
    axes[i, 0].text(-0.1, 0.5, nombre_imagen, transform=axes[i, 0].transAxes, 
                    fontsize=16, fontweight='bold', ha='right', va='center')
    
    # Escala de grises y Canales
    axes[i, 1].imshow(gray, cmap='gray')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(roi_R)
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(roi_G)
    axes[i, 3].axis('off')
    
    axes[i, 4].imshow(roi_B)
    axes[i, 4].axis('off')
    
    # Títulos de columna solo en la primera fila
    if i == 0:
        for j in range(6):
            axes[i, j].set_title(titulos_columnas[j], fontsize=16, pad=15)
    
    # Histograma de los 3 canales
    for canal, color, nombre in [(0, "red", "R"), (1, "green", "G"), (2, "blue", "B")]:
        hist = cv.calcHist([roi_rgb], [canal], None, [256], [0, 256]).ravel()
        axes[i, 5].plot(hist, color=color, alpha=0.8, label=nombre, linewidth=2)
        
    if i == 0: 
        axes[i, 5].legend(fontsize='12', loc='upper right')
        
    axes[i, 5].set_xlim([0, 256])
    axes[i, 5].set_yticks([]) # Ocultar eje Y
    axes[i, 5].grid(alpha=0.3)

# 4. Ajustar márgenes para maximizar el tamaño de las imágenes
plt.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.05, wspace=0.05, hspace=0.1)

# Guardar y mostrar
plt.savefig('seleccion_cuatro_huellas_crema.png', dpi=300, bbox_inches='tight')
plt.show()