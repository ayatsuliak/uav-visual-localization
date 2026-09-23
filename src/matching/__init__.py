MATCHER_NAMES = ("LightGlue", "LoFTR")


def create_matcher(name: str, device: str | None = None, **kwargs):
    """Створює matcher за назвою (імпорт моделей — лише за потреби)."""
    key = name.lower()
    if key == "lightglue":
        from src.matching.lightglue import LightGlueMatcher
        return LightGlueMatcher(device=device, **kwargs)
    if key == "loftr":
        from src.matching.loftr import LoFTRMatcher
        return LoFTRMatcher(device=device, **kwargs)
    raise ValueError(f"Unknown matcher '{name}', expected one of {MATCHER_NAMES}")
