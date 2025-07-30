#!/usr/bin/env python3
"""
Leishmania Screener - Lanceur Version Client
Développé par BOUNAB SOUFYANE

Lance l'application en mode client pour utilisation standard
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser

def check_dependencies():
    """Vérifie et installe les dépendances si nécessaire"""
    try:
        import cv2
        import torch
        import ultralytics
        import flask
        import flask_socketio
        return True
    except ImportError:
        print("⚠️  Dépendances manquantes détectées")
        print("📦 Installation automatique en cours...")
        
        # Importer et exécuter l'installateur
        try:
            from install_dependencies import check_and_install_dependencies
            return check_and_install_dependencies()
        except ImportError:
            print("❌ Installateur non trouvé, installation manuelle requise")
            return False

def open_browser():
    """Ouvre le navigateur après un délai"""
    time.sleep(3)  # Attendre que le serveur démarre
    webbrowser.open('http://localhost:5000')

def main():
    print("=" * 60)
    print("   LEISHMANIA SCREENER - VERSION CLIENT")
    print("   Développé par BOUNAB SOUFYANE")
    print("=" * 60)
    print("🚀 Démarrage en mode client...")
    print("👤 Interface utilisateur simplifiée")
    print("📹 Détection Leishmania en temps réel")
    print("-" * 60)
    
    # Vérifier les dépendances
    if not check_dependencies():
        print("⚠️  Dépendances manquantes, démarrage en mode limité...")
    
    try:
        # Démarrer le thread pour ouvrir le navigateur
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("🌐 Ouverture automatique du navigateur...")
        print("📱 Interface: http://localhost:5000")
        print("🛑 Appuyez sur Ctrl+C pour arrêter")
        print("-" * 60)
        
        # Lancer l'application client
        os.system("python app_client.py")
    except KeyboardInterrupt:
        print("\n🛑 Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()