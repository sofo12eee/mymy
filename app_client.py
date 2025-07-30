"""
Leishmania Screener - Version Client
Développé par BOUNAB SOUFYANE

Version utilisateur avec fonctionnalités de détection essentielles
"""

import os
import cv2
import time
import base64
import threading
import numpy as np
import platform
import psutil
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from license_system import LicenseSystem
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation Flask et SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'leishmania-client-detection'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialiser le système de licence
license_system = LicenseSystem()

class ClientDetectionService:
    def __init__(self):
        self.model = None
        self.cap = None
        self.is_running = False
        self.thread = None
        self.confidence_threshold = 0.5
        self.detection_enabled = True
        self.background_masking = False
        self.current_camera = None
        self.ip_camera_url = None
        self.frame_count = 0
        self.detection_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Optimisations GPU/CPU
        self.device = 'cpu'
        self.half_precision = False
        self.inference_times = []
        self.avg_inference_time = 0
        
        # Optimisations pour réduire la latence vidéo
        self.skip_frames = 0  # Nombre de frames à ignorer entre les détections
        self.detection_interval = 5  # Traiter 1 frame sur 5 pour l'IA (moins de CPU)
        self.frame_buffer_size = 3  # Vider plus de frames du buffer pour IP webcam
        
        # Charger le modèle de détection IA
        self.load_model()
    
    def load_model(self):
        """Charge le modèle de détection personnalisé avec optimisations GPU"""
        try:
            # Essayer différents chemins pour le modèle
            model_paths = [
                'best.pt',
                'models/best.pt',
                'attached_assets/best_1751356164931.pt',
                'yolov8n.pt'  # Modèle par défaut si le custom n'est pas trouvé
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    logger.info(f"Chargement du modèle: {model_path}")
                    
                    # Utiliser ultralytics si disponible
                    try:
                        from ultralytics import YOLO
                        import torch
                        
                        # Détection et configuration GPU
                        if torch.cuda.is_available():
                            self.device = 'cuda'
                            # Optimisations CUDA
                            torch.backends.cudnn.benchmark = True
                            torch.backends.cudnn.deterministic = False
                            logger.info("✓ GPU CUDA détecté et optimisé")
                        else:
                            self.device = 'cpu'
                            logger.info("CPU utilisé (GPU non disponible)")
                        
                        self.model = YOLO(model_path)
                        
                        # Optimisations GPU si disponible
                        if self.device == 'cuda':
                            try:
                                # Essayer la demi-précision (FP16)
                                self.model.model.half()
                                self.half_precision = True
                                logger.info("✓ Demi-précision (FP16) activée")
                            except Exception as e:
                                logger.warning(f"FP16 non supporté, utilisation FP32: {e}")
                                self.half_precision = False
                        
                        # Réchauffage du modèle
                        self._warmup_model()
                        
                        logger.info(f"✓ Modèle de détection chargé: {model_path}")
                        return True
                        
                    except ImportError:
                        logger.warning("Ultralytics non disponible, mode démo activé")
                        self.model = None
                        return False
                        
            logger.warning("Modèle personnalisé non trouvé, mode démo activé")
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def _warmup_model(self):
        """Réchauffe le modèle pour des inférences plus rapides"""
        if self.model and self.device == 'cuda':
            try:
                import torch
                logger.info("Réchauffage du modèle GPU...")
                
                # Créer une image de test
                dummy_image = torch.randn(1, 3, 640, 640).to(self.device)
                if self.half_precision:
                    dummy_image = dummy_image.half()
                
                # Effectuer quelques inférences de réchauffage
                for i in range(3):
                    _ = self.model.predict(dummy_image, verbose=False)
                
                logger.info("✓ Modèle réchauffé")
                
            except Exception as e:
                logger.warning(f"Erreur lors du réchauffage: {e}")
    
    def get_available_cameras(self):
        """Détecte les caméras disponibles"""
        cameras = []
        
        # Caméra IP Android
        cameras.append({
            'id': 'ip_camera',
            'name': 'Caméra IP (Android)',
            'type': 'ip'
        })
        
        # Caméras USB/intégrées
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append({
                        'id': str(i),
                        'name': f'Caméra USB {i}',
                        'type': 'usb'
                    })
                    cap.release()
            except:
                pass
        
        return cameras
    
    def start_detection(self, camera_id, ip_url=None, confidence=0.5):
        """Démarre la détection avec la caméra spécifiée"""
        if self.is_running:
            self.stop_detection()
        
        try:
            self.confidence_threshold = confidence
            self.current_camera = camera_id
            self.ip_camera_url = ip_url
            
            # Initialiser la capture vidéo
            if camera_id == 'ip_camera' and ip_url:
                # Construire l'URL complète pour la caméra IP
                if not ip_url.startswith('http'):
                    ip_url = f'http://{ip_url}'
                if not ip_url.endswith('/video'):
                    ip_url = f'{ip_url}/video'
                
                logger.info(f"Connexion à la caméra IP: {ip_url}")
                
                # Essayer différentes configurations pour IP Webcam
                try:
                    # Configuration principale avec buffer réduit
                    self.cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test rapide de lecture
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        logger.warning("Échec de lecture avec FFMPEG, essai avec configuration par défaut")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(ip_url)
                        
                        # Test avec configuration par défaut
                        ret, test_frame = self.cap.read()
                        if not ret or test_frame is None:
                            raise Exception("Impossible de lire les frames de la caméra IP")
                    
                    logger.info("✓ Caméra IP configurée avec succès")
                    
                except Exception as e:
                    logger.error(f"Erreur configuration IP: {e}")
                    # Essayer une dernière fois avec configuration basique
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(ip_url)
                    
            else:
                logger.info(f"Connexion à la caméra locale: {camera_id}")
                self.cap = cv2.VideoCapture(int(camera_id))
            
            if not self.cap.isOpened():
                raise Exception("Impossible d'ouvrir la caméra")
            
            # Configurer la caméra pour réduire la latence
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Optimisations spécifiques pour IP Webcam
                if camera_id == 'ip_camera':
                    # Réduire le buffer pour diminuer la latence
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Timeout plus court pour éviter les blocages
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                    logger.info("✓ Optimisations IP Webcam activées (buffer réduit)")
                    
            except Exception as e:
                logger.warning(f"Erreur configuration propriétés: {e}")
                # Continuer même si la configuration échoue
                pass
            
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.detection_count = 0
            
            # Démarrer le thread de détection
            self.thread = threading.Thread(target=self._detection_loop)
            self.thread.daemon = True
            self.thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage: {e}")
            return False
    
    def stop_detection(self):
        """Arrête la détection"""
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _detection_loop(self):
        """Boucle principale de détection optimisée pour réduire la latence"""
        last_detections = []  # Garder les dernières détections pour réutilisation
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running:
            try:
                # Lire plusieurs frames pour vider le buffer et réduire la latence
                ret, frame = None, None
                
                # Pour IP webcam, vider aggressivement le buffer
                if self.current_camera == 'ip_camera':
                    # Vider tout le buffer accumulé
                    frames_discarded = 0
                    for _ in range(self.frame_buffer_size):
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        frames_discarded += 1
                    
                    # Forcer la lecture de la frame la plus récente
                    if frames_discarded > 0:
                        # Lire une frame supplémentaire pour être sûr d'avoir la plus récente
                        ret, frame = self.cap.read()
                else:
                    # Pour USB, une seule lecture
                    ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Impossible de lire la frame ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Trop d'échecs de lecture consécutifs, arrêt de la détection")
                        socketio.emit('detection_error', {'error': 'Connexion caméra perdue'})
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Réinitialiser le compteur d'échecs
                consecutive_failures = 0
                
                self.frame_count += 1
                
                # Calculer le FPS
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0:
                    self.fps = self.frame_count / elapsed_time
                
                # Effectuer la détection seulement sur certaines frames
                detections = last_detections  # Utiliser les dernières détections par défaut
                processing_time = 0
                
                should_detect = (self.frame_count % self.detection_interval == 0)
                
                if self.detection_enabled and self.model and should_detect:
                    start_time = time.time()
                    detections = self._process_frame(frame)
                    processing_time = (time.time() - start_time) * 1000
                    last_detections = detections  # Sauvegarder pour les prochaines frames
                
                # Appliquer le masquage d'arrière-plan si activé
                if self.background_masking:
                    annotated_frame = self._apply_background_masking(frame, detections)
                else:
                    # Dessiner les détections sur la frame
                    annotated_frame = self._draw_detections(frame, detections)
                
                # Encoder avec une qualité très optimisée pour la vitesse
                if self.current_camera == 'ip_camera':
                    # Qualité plus basse pour IP webcam pour réduire la latence
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    # Redimensionner pour réduire la taille des données
                    h, w = annotated_frame.shape[:2]
                    if w > 480:  # Redimensionner si trop grand
                        scale = 480 / w
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        annotated_frame = cv2.resize(annotated_frame, (new_w, new_h))
                else:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                
                _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Envoyer les données via WebSocket
                socketio.emit('frame_update', {
                    'frame': frame_base64,
                    'fps': round(self.fps, 1),
                    'detections': len(detections),
                    'processing_time': round(processing_time, 1),
                    'total_detections': self.detection_count,
                    'detection_active': should_detect
                })
                
                # Réduire drastiquement le délai pour les caméras IP
                if self.current_camera == 'ip_camera':
                    time.sleep(0.005)  # 5ms pour IP webcam (très rapide)
                else:
                    time.sleep(0.033)  # 33ms pour caméras USB
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de détection: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    logger.error("Trop d'erreurs consécutives, arrêt de la détection")
                    socketio.emit('detection_error', {'error': f'Erreur détection: {str(e)}'})
                    break
                
                time.sleep(0.1)
    
    def _process_frame(self, frame):
        """Traite une frame avec le modèle de détection optimisé"""
        detections = []
        
        try:
            if hasattr(self.model, 'predict'):
                # Redimensionnement intelligent pour optimiser l'inférence
                h, w = frame.shape[:2]
                target_size = 640
                
                if max(h, w) > target_size:
                    # Redimensionner en conservant les proportions
                    scale = target_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                else:
                    resized_frame = frame
                    scale = 1.0
                
                # Utiliser ultralytics YOLO avec optimisations
                if self.half_precision:
                    results = self.model.predict(
                        resized_frame, 
                        conf=self.confidence_threshold, 
                        verbose=False,
                        half=True,
                        device=self.device
                    )
                else:
                    results = self.model.predict(
                        resized_frame, 
                        conf=self.confidence_threshold, 
                        verbose=False,
                        device=self.device
                    )
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Redimensionner les coordonnées si nécessaire
                            if scale != 1.0:
                                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                            
                            # Obtenir le nom de la classe
                            class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f'Class_{class_id}'
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_name': class_name,
                                'class_id': class_id
                            })
                            
                            self.detection_count += 1
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la frame: {e}")
        
        return detections
    
    def _draw_detections(self, frame, detections):
        """Dessine les détections sur la frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Dessiner la boîte englobante
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Dessiner le label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def _apply_background_masking(self, frame, detections):
        """Applique le masquage d'arrière-plan en gardant seulement les détections"""
        if not detections:
            # Si aucune détection, retourner une frame noire
            masked_frame = np.zeros_like(frame)
            return masked_frame
        
        # Créer un masque pour les détections
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Remplir le masque avec les zones détectées
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Agrandir légèrement la zone pour un meilleur effet visuel
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Créer une zone masquée circulaire/elliptique pour un effet plus naturel
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius_x = (x2 - x1) // 2
            radius_y = (y2 - y1) // 2
            
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
        
        # Créer une frame noire
        masked_frame = np.zeros_like(frame)
        
        # Appliquer le masque pour garder seulement les zones détectées
        masked_frame[mask > 0] = frame[mask > 0]
        
        # Dessiner les détections sur la frame masquée
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Dessiner la boîte englobante avec une couleur plus vive
            cv2.rectangle(masked_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Dessiner le label avec fond semi-transparent
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Fond du label
            cv2.rectangle(masked_frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0] + 10, bbox[1]), (0, 255, 0), -1)
            
            # Texte du label
            cv2.putText(masked_frame, label, (bbox[0] + 5, bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return masked_frame
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        if self.model:
            try:
                if hasattr(self.model, 'names'):
                    return {
                        'status': 'Chargé',
                        'type': 'Leishmania Screener',
                        'classes': list(self.model.names.values()),
                        'num_classes': len(self.model.names)
                    }
                else:
                    return {
                        'status': 'Chargé',
                        'type': 'OpenCV DNN',
                        'classes': [],
                        'num_classes': 0
                    }
            except:
                return {
                    'status': 'Chargé (informations limitées)',
                    'type': 'Modèle personnalisé',
                    'classes': [],
                    'num_classes': 0
                }
        else:
            return {
                'status': 'Non chargé',
                'type': 'Aucun',
                'classes': [],
                'num_classes': 0
            }
    
    def get_gpu_info(self):
        """Retourne les informations GPU ou CPU selon disponibilité"""
        try:
            # Vérifier si PyTorch avec CUDA est disponible
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    # Informations mémoire détaillées
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    return {
                        'device': gpu_name,
                        'type': 'GPU',
                        'memory_gb': round(gpu_memory, 1),
                        'memory_used_gb': round(gpu_memory_used, 1),
                        'memory_cached_gb': round(gpu_memory_cached, 1),
                        'cuda_version': torch.version.cuda,
                        'available': True,
                        'using_gpu': True,
                        'half_precision': getattr(self, 'half_precision', False),
                        'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'optimized': True
                    }
            except ImportError:
                pass
            
            # Si pas de GPU, utiliser CPU
            cpu_info = platform.processor()
            
            # Si platform.processor() est vide, utiliser des infos alternatives
            if not cpu_info or cpu_info.strip() == "":
                try:
                    # Essayer de lire /proc/cpuinfo sur Linux
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if line.startswith('model name'):
                                cpu_info = line.split(':')[1].strip()
                                break
                except:
                    # Sur Windows, utiliser des infos alternatives
                    import subprocess
                    try:
                        result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                                capture_output=True, text=True)
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            cpu_info = lines[1].strip()
                    except:
                        cpu_info = f"Processeur {platform.machine()}"
            
            return {
                'device': cpu_info or 'Processeur non identifié',
                'type': 'CPU',
                'cores': psutil.cpu_count(logical=True),
                'available': True,
                'using_gpu': False
            }
            
        except Exception as e:
            return {
                'device': 'Périphérique non identifié',
                'type': 'CPU',
                'available': False,
                'using_gpu': False
            }

# Instance globale du service de détection
detection_service = ClientDetectionService()

# Fonction de vérification de licence
def check_license():
    """Vérifie la licence au démarrage"""
    license_info = license_system.get_license_info()
    return license_info

# Routes Flask
@app.route('/')
def index():
    # Vérifier la licence
    license_info = check_license()
    
    if not license_info["valid"]:
        return redirect(url_for('license_page'))
    
    return render_template('client.html', license_info=license_info)

@app.route('/license')
def license_page():
    """Page de gestion des licences"""
    license_info = check_license()
    return render_template('license.html', license_info=license_info)

@app.route('/license', methods=['POST'])
def activate_license():
    """Activer une licence"""
    license_key = request.form.get('license_key', '').strip()
    
    if not license_key:
        flash('Veuillez entrer une clé de licence', 'error')
        return redirect(url_for('license_page'))
    
    # Vérifier la licence
    verification = license_system.verify_license(license_key)
    
    if verification["valid"]:
        # Sauvegarder la licence
        if license_system.save_license(license_key):
            flash(f'Licence activée avec succès! Expire le {verification["expiry_date"][:10]}', 'success')
            return redirect(url_for('index'))
        else:
            flash('Erreur lors de la sauvegarde de la licence', 'error')
    else:
        flash(f'Licence invalide: {verification["error"]}', 'error')
    
    return redirect(url_for('license_page'))

@app.route('/license/info')
def license_info():
    """API pour les informations de licence"""
    license_info = check_license()
    return jsonify(license_info)

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'running',
        'model_loaded': detection_service.model is not None,
        'is_detecting': detection_service.is_running,
        'mode': 'client'
    })

@app.route('/api/gpu-info')
def api_gpu_info():
    """API pour récupérer les informations GPU/CPU"""
    return jsonify(detection_service.get_gpu_info())

# Événements SocketIO
@socketio.on('connect')
def handle_connect():
    logger.info('Client connecté')
    
    # Envoyer les informations de licence
    license_info = check_license()
    
    emit('connected', {'status': 'connected'})
    emit('license_info', license_info)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client déconnecté')

@socketio.on('get_cameras')
def handle_get_cameras():
    cameras = detection_service.get_available_cameras()
    emit('cameras_list', {'cameras': cameras})

@socketio.on('start_detection')
def handle_start_detection(data):
    camera_id = data.get('camera_id')
    ip_url = data.get('ip_url')
    confidence = data.get('confidence', 0.5)
    
    success = detection_service.start_detection(camera_id, ip_url, confidence)
    emit('detection_started', {'success': success, 'camera_id': camera_id})

@socketio.on('stop_detection')
def handle_stop_detection():
    detection_service.stop_detection()
    emit('detection_stopped', {'success': True})

@socketio.on('get_model_info')
def handle_get_model_info():
    model_info = detection_service.get_model_info()
    emit('model_info', model_info)

@socketio.on('update_settings')
def handle_update_settings(data):
    if 'confidence' in data:
        detection_service.confidence_threshold = data['confidence']
        logger.info(f"Seuil de confiance mis à jour: {data['confidence']}")
    if 'detection_enabled' in data:
        detection_service.detection_enabled = data['detection_enabled']
        logger.info(f"Détection activée/désactivée: {data['detection_enabled']}")
    if 'background_masking' in data:
        detection_service.background_masking = data['background_masking']
        logger.info(f"Masquage arrière-plan: {data['background_masking']}")
    
    emit('settings_updated', {'success': True})

if __name__ == '__main__':
    print("=" * 60)
    print("   LEISHMANIA SCREENER - Version Client")
    print("   Développé par BOUNAB SOUFYANE")
    print("=" * 60)
    print("🚀 Démarrage de l'application...")
    print("🌐 Interface web: http://localhost:5000")
    print("📹 Détection temps réel avec LEISHMANIA SCREENER")
    print("👤 Mode utilisateur")
    print("-" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)