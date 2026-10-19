"""Re-export of Member B's Stage 4 for A's API (app/b_interface.py imports `validity.get_validity`)."""
from stage4_validity import get_validity

__all__ = ["get_validity"]
