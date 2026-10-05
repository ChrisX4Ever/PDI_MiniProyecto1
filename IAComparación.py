import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

# ==========================================
# 1. Pipeline del Grupo (Manual / Morfológico)
# ==========================================
def segmentacion_grupo(img_gray, operacion='estiramiento'):
    # A. Realce
    if operacion == 'ecualizacion':
        img_realce = cv2.equalizeHist(img_gray)
    else: # Estiramiento
        r_a, r_b = img_gray.min(), img_gray.max()
        if r_a == r_b:
            img_realce = img_gray
        else:
            img_realce = np.clip((img_gray.astype(np.float32) - r_a) * 255 / (r_b - r_a), 0, 255).astype(np.uint8)

    # B. Máscara inicial con Otsu invertido (crestas en blanco)
    _, mascara_inicial = cv2.threshold(img_realce, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # C. Refinamiento Morfológico (Apertura y Cierre)
    kernel = np.ones((3, 3), np.uint8)
    masc_apertura = cv2.morphologyEx(mascara_inicial, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara_final = cv2.morphologyEx(masc_apertura, cv2.MORPH_CLOSE, kernel, iterations=1)

    # D. Imagen resultado
    img_resultado = cv2.bitwise_and(img_gray, img_gray, mask=mascara_final)
    
    return mascara_final, img_resultado

# ==========================================
# 2. Pipeline de IA (K-Means)
# ==========================================
def segmentacion_ia(img_gray):
    pixeles = img_gray.reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(pixeles)
    
    mascara_ia = kmeans.labels_.reshape(img_gray.shape)
    
    centroides = kmeans.cluster_centers_.flatten()
    if centroides[0] > centroides[1]:
        mascara_ia = 1 - mascara_ia
        
    mascara_ia = (mascara_ia * 255).astype(np.uint8)
    img_resultado = cv2.bitwise_and(img_gray, img_gray, mask=mascara_ia)
    
    return mascara_ia, img_resultado

# ==========================================
# 3. Función de Cálculo de Métricas
# ==========================================
def calcular_metricas_comparativas(mascara_grupo, mascara_ia, img_original, img_resultado_grupo):
    m_grupo = (mascara_grupo > 127).astype(bool)
    m_ia = (mascara_ia > 127).astype(bool)
    
    # Métricas de Máscara
    interseccion = np.logical_and(m_grupo, m_ia).sum()
    union = np.logical_or(m_grupo, m_ia).sum()
    
    iou = interseccion / union if union != 0 else 0
    dice = (2. * interseccion) / (m_grupo.sum() + m_ia.sum()) if (m_grupo.sum() + m_ia.sum()) != 0 else 0
    
    # Métricas Globales
    img_resultado_ia = cv2.bitwise_and(img_original, img_original, mask=mascara_ia)
    mae = mean_absolute_error(img_resultado_grupo.flatten(), img_resultado_ia.flatten())
    
    mse = np.mean((img_resultado_grupo.astype(np.float64) - img_resultado_ia.astype(np.float64)) ** 2)
    psnr = 100 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
    
    return iou, dice, mae, psnr, img_resultado_ia

# ==========================================
# 4. Ejecución y Visualización
# ==========================================
ruta_foto = "Fotos_PDI_2026_Grupo1/d1_crema.jpg"

img_bgr = cv2.imread(ruta_foto)
if img_bgr is None:
    print(f"Error: No se encontró la imagen '{ruta_foto}'.")
else:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Generar salidas de ambos métodos
    masc_grupo, res_grupo = segmentacion_grupo(img_gray, operacion='estiramiento')
    masc_ia, _ = segmentacion_ia(img_gray)
    
    # Calcular métricas
    iou, dice, mae, psnr, res_ia = calcular_metricas_comparativas(masc_grupo, masc_ia, img_gray, res_grupo)
    
    # Mostrar resultados en la terminal
    print("\n--- RESULTADOS DE LA COMPARACIÓN ---")
    print(f"Métricas de Máscara:  IoU = {iou:.4f}  |  Dice = {dice:.4f}")
    print(f"Métricas Globales:   MAE = {mae:.2f}    |  PSNR = {psnr:.2f} dB\n")
    
    # Visualización gráfica lado a lado
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'Comparación Procedimiento del Grupo vs IA\nIoU: {iou:.3f} | Dice: {dice:.3f} | MAE: {mae:.2f} | PSNR: {psnr:.2f} dB', fontsize=14)
    
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('1. Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(masc_grupo, cmap='gray')
    axes[0, 1].set_title('2. Máscara Grupo (Otsu + Morfología)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(res_grupo, cmap='gray')
    axes[0, 2].set_title('3. Resultado Grupo')
    axes[0, 2].axis('off')
    
    axes[1, 0].axis('off') # Espacio en blanco para alinear
    
    axes[1, 1].imshow(masc_ia, cmap='gray')
    axes[1, 1].set_title('4. Máscara IA (K-Means)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(res_ia, cmap='gray')
    axes[1, 2].set_title('5. Resultado IA')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()