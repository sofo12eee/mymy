#!/usr/bin/env python3
"""
Installation automatique des dépendances - Leishmania Screener
Développé par BOUNAB SOUFYANE
"""

import subprocess
import sys
import os

def install_package(package):
    """Installe un package Python"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Vérifie et installe les dépendances nécessaires"""
    print("=" * 60)
    print("   INSTALLATION DES DÉPENDANCES")
    print("   Leishmania Screener - BOUNAB SOUFYANE")
    print("=" * 60)
    
    # Liste des dépendances requises
    dependencies = [
        'flask',
        'flask-socketio',
        'opencv-python',
        'torch',
        'ultralytics',
        'numpy',
        'psutil'
    ]
    
    installed = []
    failed = []
    
    for dep in dependencies:
        print(f"📦 Installation de {dep}...")
        if install_package(dep):
            installed.append(dep)
            print(f"✅ {dep} installé avec succès")
        else:
            failed.append(dep)
            print(f"❌ Échec installation {dep}")
    
    print("\n" + "=" * 60)
    if len(installed) == len(dependencies):
        print("✅ TOUTES LES DÉPENDANCES INSTALLÉES AVEC SUCCÈS")
        print("🚀 Prêt pour le démarrage de l'application")
        return True
    else:
        print("⚠️  CERTAINES DÉPENDANCES ONT ÉCHOUÉ")
        print(f"✅ Installées: {', '.join(installed)}")
        print(f"❌ Échouées: {', '.join(failed)}")
        print("📝 L'application fonctionnera en mode démo limité")
        return False

if __name__ == "__main__":
    check_and_install_dependencies()