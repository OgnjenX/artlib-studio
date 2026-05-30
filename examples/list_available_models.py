import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARL_PATH = os.path.join(ROOT, "AdaptiveResonanceLib")
STUDIO_PATH = os.path.join(ROOT, "artlib-studio")
if ARL_PATH not in sys.path:
    sys.path.insert(0, ARL_PATH)
if STUDIO_PATH not in sys.path:
    sys.path.insert(0, STUDIO_PATH)

from artlib_studio.core.registry import list_adapters

def main():
    print("Available ARTLib Studio adapters:\n")
    adapters = list_adapters()
    for adapter in adapters:
        print(f"* {adapter.name}")
        print(f"  key: {adapter.model_key}")
        print("  capabilities:")
        for cap in adapter.capabilities:
            print(f"  * {cap.name.lower()}")
        print()

if __name__ == "__main__":
    main()
