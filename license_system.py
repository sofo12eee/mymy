"""
Système de Licence Annuelle - Leishmania Screener
Développé par BOUNAB SOUFYANE

Génération et vérification de licences chiffrées
"""

import os
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
import platform
import uuid

class LicenseSystem:
    def __init__(self):
        # Clé secrète pour le chiffrement (à garder confidentielle)
        self.secret_key = "LEISHMANIA_SCREENER_SECRET_2025_BOUNAB_SOUFYANE"
        self.license_file = "license.dat"
    
    def _simple_encrypt(self, data):
        """Chiffrement XOR simple"""
        key = self.secret_key
        result = ""
        for i, char in enumerate(data):
            result += chr(ord(char) ^ ord(key[i % len(key)]))
        return result
    
    def _simple_decrypt(self, encrypted_data):
        """Déchiffrement XOR simple"""
        return self._simple_encrypt(encrypted_data)  # XOR est symétrique
        
    def generate_machine_id(self):
        """Génère un ID unique basé sur le matériel"""
        try:
            # Combinaison d'identifiants matériel
            hostname = platform.node()
            system = platform.system()
            processor = platform.processor()
            
            # Créer un hash unique
            machine_data = f"{hostname}-{system}-{processor}"
            machine_id = hashlib.sha256(machine_data.encode()).hexdigest()[:16]
            return machine_id
        except:
            # Fallback si erreur
            return str(uuid.uuid4())[:16]
    
    def create_license(self, user_name, expiry_date, license_type="standard"):
        """Crée une licence chiffrée"""
        try:
            machine_id = self.generate_machine_id()
            
            # Données de la licence
            license_data = {
                "user_name": user_name,
                "expiry_date": expiry_date,
                "license_type": license_type,
                "machine_id": machine_id,
                "created_date": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Convertir en JSON
            license_json = json.dumps(license_data, sort_keys=True)
            
            # Créer signature HMAC
            signature = hmac.new(
                self.secret_key.encode(),
                license_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Données finales avec signature
            final_data = {
                "license": license_data,
                "signature": signature
            }
            
            # Chiffrement simple XOR avec base64
            final_json = json.dumps(final_data)
            encrypted_data = self._simple_encrypt(final_json)
            
            # Encoder en base64 pour stockage
            license_key = base64.b64encode(encrypted_data.encode()).decode()
            
            return {
                "success": True,
                "license_key": license_key,
                "machine_id": machine_id,
                "expiry_date": expiry_date
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def verify_license(self, license_key=None):
        """Vérifie une licence"""
        try:
            # Lire depuis fichier ou paramètre
            if license_key is None:
                if not os.path.exists(self.license_file):
                    return {
                        "valid": False,
                        "error": "Aucune licence trouvée",
                        "expired": False
                    }
                
                with open(self.license_file, 'r') as f:
                    license_key = f.read().strip()
            
            # Décoder et déchiffrer avec gestion d'erreurs
            try:
                # Nettoyage de la clé
                license_key = license_key.strip()
                
                # Décoder base64
                encrypted_data = base64.b64decode(license_key).decode('utf-8')
                
                # Déchiffrer
                decrypted_data = self._simple_decrypt(encrypted_data)
                
                # Parser JSON
                license_info = json.loads(decrypted_data)
                
            except Exception as decode_error:
                return {
                    "valid": False,
                    "error": f"Erreur décodage licence: {str(decode_error)}",
                    "expired": False
                }
            
            # Vérifier signature
            license_data = license_info["license"]
            provided_signature = license_info["signature"]
            
            # Recalculer signature
            license_json = json.dumps(license_data, sort_keys=True)
            expected_signature = hmac.new(
                self.secret_key.encode(),
                license_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if provided_signature != expected_signature:
                return {
                    "valid": False,
                    "error": "Licence corrompue ou falsifiée",
                    "expired": False
                }
            
            # Vérifier ID machine (accepter les licences universelles)
            current_machine_id = self.generate_machine_id()
            universal_ids = ["GENERIC_MACHINE_ID", "UNIVERSAL_LICENSE", "CLIENT_UNIVERSAL"]
            
            # Accepter licence universelle OU licence spécifique à cette machine
            if (license_data["machine_id"] not in universal_ids and 
                license_data["machine_id"] != current_machine_id):
                return {
                    "valid": False,
                    "error": "Licence non valide pour cette machine",
                    "expired": False
                }
            
            # Vérifier expiration
            expiry_key = "expires_at" if "expires_at" in license_data else "expiry_date"
            expiry_date = datetime.fromisoformat(license_data[expiry_key])
            current_date = datetime.now()
            
            if current_date > expiry_date:
                return {
                    "valid": False,
                    "error": "Licence expirée",
                    "expired": True,
                    "expiry_date": license_data.get("expires_at", license_data.get("expiry_date", ""))
                }
            
            # Licence valide
            days_remaining = (expiry_date - current_date).days
            
            return {
                "valid": True,
                "license_data": license_data,
                "days_remaining": days_remaining,
                "expiry_date": license_data.get("expires_at", license_data.get("expiry_date", "")),
                "user_name": license_data["user_name"],
                "license_type": license_data["license_type"]
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Erreur vérification licence: {str(e)}",
                "expired": False
            }
    
    def save_license(self, license_key):
        """Sauvegarde une licence dans un fichier"""
        try:
            with open(self.license_file, 'w') as f:
                f.write(license_key)
            return True
        except:
            return False
    
    def get_license_info(self):
        """Retourne les informations de licence actuelles"""
        return self.verify_license()

# Fonction utilitaire pour générer une licence
def generate_license_for_user(user_name, duration_months=12):
    """Génère une licence pour un utilisateur"""
    license_system = LicenseSystem()
    
    # Calculer date d'expiration
    expiry_date = datetime.now() + timedelta(days=duration_months * 30)
    expiry_str = expiry_date.isoformat()
    
    result = license_system.create_license(user_name, expiry_str)
    
    if result["success"]:
        print(f"✅ Licence générée pour {user_name}")
        print(f"📅 Expire le: {expiry_date.strftime('%d/%m/%Y')}")
        print(f"🔑 Clé: {result['license_key']}")
        print(f"🖥️  ID Machine: {result['machine_id']}")
        return result['license_key']
    else:
        print(f"❌ Erreur: {result['error']}")
        return None

if __name__ == "__main__":
    # Test du système
    print("=== Test Système de Licence ===")
    
    # Générer une licence de test
    test_license = generate_license_for_user("Utilisateur Test", 12)
    
    if test_license:
        # Tester la vérification
        license_system = LicenseSystem()
        verification = license_system.verify_license(test_license)
        
        if verification["valid"]:
            print(f"✅ Licence valide - {verification['days_remaining']} jours restants")
        else:
            print(f"❌ Licence invalide: {verification['error']}")