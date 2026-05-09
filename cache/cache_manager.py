"""
Cache JSON avec expiration (TTL) pour les données externes.

Évite de refaire des appels API ou des calculs coûteux inutilement.
Chaque entrée stocke un timestamp d'expiration ; une entrée expirée est
automatiquement supprimée au prochain accès.

Utilisation :
    from cache.cache_manager import CacheManager
    cache = CacheManager()              # utilise le répertoire cache/ par défaut
    cache.set("osm_paris", data)        # TTL = 7 jours par défaut
    data = cache.get("osm_paris")       # None si absent ou expiré
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Répertoire par défaut : le dossier cache/ à côté de ce fichier
_DEFAULT_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))

# TTL par type de données (en heures)
TTL_CITY_DATA = 168      # 7 jours – données ville OpenTripMap
TTL_OSM_DATA = 168       # 7 jours – données Overpass / OSM
TTL_TRAVEL_MATRIX = 720  # 30 jours – matrice OSRM (très stable)
TTL_SHORT = 24           # 1 jour – données volatiles


class CacheManager:
    """Gestionnaire de cache JSON avec TTL par entrée."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── Lecture ──────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """
        Retourne les données mises en cache, ou None si absentes / expirées.
        Supprime automatiquement l'entrée expirée.
        """
        path = self._path(key)
        if not os.path.exists(path):
            return None

        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)

            if time.time() > entry.get("expires_at", 0):
                logger.debug("Cache expiré pour '%s' – suppression", key)
                self._delete(path)
                return None

            logger.debug("Cache hit pour '%s' (expire dans %.1fh)",
                         key, (entry["expires_at"] - time.time()) / 3600)
            return entry["data"]

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Cache corrompu pour '%s' : %s – suppression", key, e)
            self._delete(path)
            return None

    # ── Écriture ─────────────────────────────────────────────────────────

    def set(self, key: str, data: Any, ttl_hours: int = TTL_CITY_DATA) -> None:
        """Enregistre les données avec un TTL (en heures)."""
        entry = {
            "key": key,
            "created_at": time.time(),
            "expires_at": time.time() + ttl_hours * 3600,
            "ttl_hours": ttl_hours,
            "data": data,
        }
        path = self._path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            logger.debug("Cache écrit pour '%s' (TTL %dh)", key, ttl_hours)
        except OSError as e:
            logger.warning("Impossible d'écrire le cache pour '%s' : %s", key, e)

    # ── Invalidation ─────────────────────────────────────────────────────

    def invalidate(self, key: str) -> None:
        """Supprime une entrée du cache."""
        self._delete(self._path(key))

    def invalidate_all(self) -> int:
        """Supprime toutes les entrées du cache. Retourne le nombre supprimé."""
        removed = 0
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".cache.json"):
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                    removed += 1
                except OSError:
                    pass
        logger.info("%d entrées de cache supprimées", removed)
        return removed

    def purge_expired(self) -> int:
        """Supprime les entrées expirées. Retourne le nombre supprimé."""
        removed = 0
        now = time.time()
        for fname in os.listdir(self.cache_dir):
            if not fname.endswith(".cache.json"):
                continue
            path = os.path.join(self.cache_dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    entry = json.load(f)
                if now > entry.get("expires_at", 0):
                    os.remove(path)
                    removed += 1
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        if removed:
            logger.info("%d entrées de cache expirées purgées", removed)
        return removed

    def info(self, key: str) -> Optional[dict]:
        """Retourne les métadonnées d'une entrée sans les données."""
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
            now = time.time()
            return {
                "key": key,
                "created_at": entry.get("created_at"),
                "expires_at": entry.get("expires_at"),
                "ttl_hours": entry.get("ttl_hours"),
                "remaining_hours": max(0, (entry["expires_at"] - now) / 3600),
                "expired": now > entry.get("expires_at", 0),
            }
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    # ── Interne ──────────────────────────────────────────────────────────

    def _path(self, key: str) -> str:
        safe = (
            key.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace(":", "_")
        )
        return os.path.join(self.cache_dir, f"{safe}.cache.json")

    @staticmethod
    def _delete(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


# Instance globale (singleton léger)
_default_cache: Optional[CacheManager] = None


def get_default_cache() -> CacheManager:
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheManager()
    return _default_cache
