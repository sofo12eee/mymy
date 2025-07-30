# Guide d'Optimisation GPU - Réduction de Latence

## Problème Identifié
Latence de détection élevée (130ms) malgré la présence d'un GPU.

## Optimisations Implémentées

### 1. Configuration GPU Optimisée
```python
# Activation optimisations CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Demi-précision (FP16) si supportée
self.model.model.half()
self.half_precision = True
```

### 2. Réchauffage du Modèle
- 3 inférences de réchauffage au démarrage
- Élimine les latences de première inférence
- Optimise les caches GPU

### 3. Redimensionnement Intelligent
```python
# Redimensionner les frames pour optimiser l'inférence
target_size = 640  # Taille optimale YOLOv8
if max(h, w) > target_size:
    scale = target_size / max(h, w)
    resized_frame = cv2.resize(frame, (new_w, new_h))
```

### 4. Paramètres d'Inférence Optimisés
```python
if self.half_precision:
    results = self.model.predict(
        resized_frame, 
        conf=self.confidence_threshold, 
        verbose=False,
        half=True,
        device=self.device
    )
```

## Résultats Attendus

### Avant Optimisation
- **Latence :** 130ms
- **GPU :** Sous-utilisé
- **Mémoire :** Non optimisée
- **Précision :** FP32

### Après Optimisation
- **Latence :** 30-50ms (réduction 60-75%)
- **GPU :** Pleinement utilisé
- **Mémoire :** FP16 si supporté
- **Throughput :** 2-3x plus rapide

## Vérifications

### 1. Utilisation GPU
```python
# Vérifier que le modèle est sur GPU
print(f"Device: {self.device}")
print(f"Half precision: {self.half_precision}")
print(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
```

### 2. Monitoring Performance
- Temps d'inférence affiché en temps réel
- Statistiques moyennes sur 10 frames
- Utilisation mémoire GPU détaillée

### 3. Interface Utilisateur
- Indicateur GPU/CPU dans l'interface
- Temps d'inférence moyen affiché
- Mode demi-précision indiqué

## Troubleshooting

### Si GPU non détecté
1. Vérifier PyTorch avec CUDA : `torch.cuda.is_available()`
2. Vérifier drivers NVIDIA
3. Réinstaller PyTorch avec CUDA

### Si latence toujours élevée
1. Vérifier la taille des frames d'entrée
2. Réduire la résolution de capture
3. Augmenter le buffer de capture : `CAP_PROP_BUFFERSIZE = 1`

### Si erreurs FP16
- Le modèle retombera automatiquement en FP32
- Certains GPU anciens ne supportent pas FP16

## Commandes de Test

```bash
# Tester l'installation GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Vérifier performances
python -c "
import torch
import time
if torch.cuda.is_available():
    x = torch.randn(1, 3, 640, 640).cuda()
    start = time.time()
    for i in range(100):
        y = torch.relu(x)
    print(f'GPU inference time: {(time.time()-start)*10:.1f}ms per frame')
"
```

## Optimisations Avancées (Optionnelles)

### TensorRT (NVIDIA)
```python
# Optimisation avec TensorRT
model.export(format='engine', half=True)
```

### Batch Processing
```python
# Traiter plusieurs frames ensemble
batch_frames = torch.stack([frame1, frame2, frame3])
results = model.predict(batch_frames)
```

### Memory Pinning
```python
# Accélérer les transferts CPU->GPU
frame_tensor = torch.from_numpy(frame).pin_memory().cuda()
```

---

**Développé par BOUNAB SOUFYANE**
**Date :** 10 Juillet 2025
**Objectif :** Réduire latence de 130ms à <50ms