import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ==========================================
# 1. Funciones Matemáticas y Derivadas
# ==========================================

def relu(Z):
    """Activación ReLU para capas ocultas."""
    return np.maximum(0, Z)

def derivada_relu(Z):
    """Derivada de ReLU (1 si Z > 0, sino 0)."""
    return (Z > 0).astype(float)

def sigmoide(Z):
    """Activación Sigmoide para la probabilidad final."""
    # Clip para evitar desbordamientos numéricos (estabilidad matemática)
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

def entropia_cruzada_binaria(Y_real, Y_pred):
    """
    Función de Riesgo Empírico para clasificación (Binary Cross-Entropy).
    Matemáticamente penaliza fuertemente predicciones seguras pero incorrectas.
    """
    m = Y_real.shape[0]
    # Epsilon minúsculo para evitar el logaritmo de cero
    eps = 1e-15
    Y_pred = np.clip(Y_pred, eps, 1 - eps)
    coste = -(1/m) * np.sum(Y_real * np.log(Y_pred) + (1 - Y_real) * np.log(1 - Y_pred))
    return coste

# ==========================================
# 2. Generación del Dataset "Two Moons"
# ==========================================
# Generamos 1000 muestras de un problema altamente no lineal
X, Y = make_moons(n_samples=1000, noise=0.15, random_state=42)
Y = Y.reshape(-1, 1) # Asegurar que Y sea un vector columna (1000 x 1)
n_muestras = X.shape[0]

# ==========================================
# 3. Inicialización de la Red (2 -> 10 -> 10 -> 1)
# ==========================================
np.random.seed(42)

# Usamos inicialización He (estándar matemático para ReLU)
W1 = np.random.randn(2, 10) * np.sqrt(2. / 2)
b1 = np.zeros((1, 10))

W2 = np.random.randn(10, 10) * np.sqrt(2. / 10)
b2 = np.zeros((1, 10))

W3 = np.random.randn(10, 1) * np.sqrt(2. / 10)
b3 = np.zeros((1, 1))

# ==========================================
# 4. Bucle de Entrenamiento (Mini-Batch SGD)
# ==========================================
tasa_aprendizaje = 0.1
epocas = 1500
tamano_batch = 32 # El tamaño de la submuestra estocástica

historial_coste = []

print("Iniciando entrenamiento de la Red Neuronal Profunda...")

for epoca in range(epocas):
    # 1. Barajar el dataset para inyectar estocasticidad
    permutacion = np.random.permutation(n_muestras)
    X_barajado = X[permutacion]
    Y_barajado = Y[permutacion]
    
    coste_epoca = 0
    num_batches = 0
    
    # 2. Iterar sobre mini-batches
    for i in range(0, n_muestras, tamano_batch):
        X_batch = X_barajado[i:i + tamano_batch]
        Y_batch = Y_barajado[i:i + tamano_batch]
        m_batch = X_batch.shape[0]
        
        # --- PASO FORWARD ---
        # Capa Oculta 1
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        
        # Capa Oculta 2
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        
        # Capa de Salida
        Z3 = np.dot(A2, W3) + b3
        A3 = sigmoide(Z3) # Predicción final
        
        # --- PASO BACKWARD (Cálculo del Gradiente) ---
        # Gradiente de la función de coste con Sigmoide final
        dZ3 = A3 - Y_batch
        dW3 = (1/m_batch) * np.dot(A2.T, dZ3)
        db3 = (1/m_batch) * np.sum(dZ3, axis=0, keepdims=True)
        
        # Propagación a Capa Oculta 2
        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * derivada_relu(Z2)
        dW2 = (1/m_batch) * np.dot(A1.T, dZ2)
        db2 = (1/m_batch) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Propagación a Capa Oculta 1
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * derivada_relu(Z1)
        dW1 = (1/m_batch) * np.dot(X_batch.T, dZ1)
        db1 = (1/m_batch) * np.sum(dZ1, axis=0, keepdims=True)
        
        # --- ACTUALIZACIÓN DE PESOS ---
        W3 -= tasa_aprendizaje * dW3
        b3 -= tasa_aprendizaje * db3
        W2 -= tasa_aprendizaje * dW2
        b2 -= tasa_aprendizaje * db2
        W1 -= tasa_aprendizaje * dW1
        b1 -= tasa_aprendizaje * db1
        
        # Acumular el coste para estadísticas
        coste_epoca += entropia_cruzada_binaria(Y_batch, A3)
        num_batches += 1
        
    # Guardar el coste promedio de la época
    historial_coste.append(coste_epoca / num_batches)
    
    if (epoca + 1) % 300 == 0:
        print(f"Época {epoca+1:4d} | Función de Coste: {historial_coste[-1]:.4f}")

# ==========================================
# 5. Visualización Académica de Resultados
# ==========================================

plt.style.use('default')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Gráfica 1: Curva de Optimización ---
ax1.plot(historial_coste, color='#d62728', linewidth=2)
ax1.set_title('Convergencia del Coste (Mini-Batch SGD)')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Entropía Cruzada Binaria')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Gráfica 2: Frontera de Decisión ---
# Crear la malla del plano
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predecir toda la malla
malla_entradas = np.c_[xx.ravel(), yy.ravel()]
Z1_m_test = np.dot(malla_entradas, W1) + b1
A1_m_test = relu(Z1_m_test)
Z2_m_test = np.dot(A1_m_test, W2) + b2
A2_m_test = relu(Z2_m_test)
Z3_m_test = np.dot(A2_m_test, W3) + b3
A3_m_test = sigmoide(Z3_m_test)
Z_pred = A3_m_test.reshape(xx.shape)

# Dibujar frontera y puntos
contour = ax2.contourf(xx, yy, Z_pred, levels=20, cmap='RdBu_r', alpha=0.8)
fig.colorbar(contour, ax=ax2, label='Probabilidad Predicha')

# Pintar puntos reales
ax2.scatter(X[Y[:,0]==0, 0], X[Y[:,0]==0, 1], c='red', edgecolors='k', label='Clase 0', alpha=0.7)
ax2.scatter(X[Y[:,0]==1, 0], X[Y[:,0]==1, 1], c='blue', edgecolors='k', label='Clase 1', alpha=0.7)

ax2.set_title('Frontera de Decisión Altamente No Lineal')
ax2.set_xlabel('Dimensión $x_1$')
ax2.set_ylabel('Dimensión $x_2$')
ax2.legend()

plt.tight_layout()
plt.savefig('dnn_moons_sgd.png', dpi=300)
plt.show()