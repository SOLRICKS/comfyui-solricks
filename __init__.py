__version__ = "0.2.1"


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


try:
    from .VideoAdaptiveAA import (
        NODE_CLASS_MAPPINGS as AA_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as AA_NAMES,
    )

    NODE_CLASS_MAPPINGS.update(AA_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(AA_NAMES)

except Exception as e:
    print(f"[SOLRICKS] VideoAdaptiveAA failed to load, skipped: {type(e).__name__}: {e}")


try:
    from .VideoTAADLAA import (
        NODE_CLASS_MAPPINGS as TAA_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as TAA_NAMES,
    )

    NODE_CLASS_MAPPINGS.update(TAA_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(TAA_NAMES)


except Exception as e:
    print(f"[SOLRICKS] VideoTAADLAA failed to load: {type(e).__name__}: {e}")


try:
    from .VideoDetailRefiner import (
        NODE_CLASS_MAPPINGS as DETAIL_REFINER_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as DETAIL_REFINER_NAMES,
    )

    NODE_CLASS_MAPPINGS.update(DETAIL_REFINER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(DETAIL_REFINER_NAMES)

except Exception as e:
    print(f"[SOLRICKS] VideoDetailRefiner failed to load: {type(e).__name__}: {e}")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]