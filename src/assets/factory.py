from functools import lru_cache

from ..config.settings import ASSET_STORAGE_BACKEND
from .store import AssetStore, create_asset_store


@lru_cache(maxsize=1)
def get_asset_store() -> AssetStore:
    return create_asset_store(ASSET_STORAGE_BACKEND)
