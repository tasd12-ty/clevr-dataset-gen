#!/usr/bin/env python3
"""
Diagnostic script for Blender import issues on Linux.
Usage: blender --background --python diagnose_blender_import.py
"""

import sys
import os

print("=" * 80)
print("Blender Python Import Diagnostics")
print("=" * 80)

# 1. Blender and Python version
try:
    import bpy
    blender_version = ".".join(map(str, bpy.app.version))
    print(f"\n✓ Blender version: {blender_version}")
except Exception as e:
    print(f"\n✗ Cannot import bpy: {e}")
    blender_version = "unknown"

print(f"✓ Python version: {sys.version}")
print(f"✓ Python executable: {sys.executable}")

# 2. Python path
print(f"\n{'Python sys.path:':-^80}")
for i, path in enumerate(sys.path, 1):
    print(f"  {i:2d}. {path}")

# 3. Site-packages locations
print(f"\n{'Site-packages locations:':-^80}")
import site
for sp in site.getsitepackages():
    exists = "✓" if os.path.exists(sp) else "✗"
    print(f"  {exists} {sp}")

    # Check for .pth files
    if os.path.exists(sp):
        pth_files = [f for f in os.listdir(sp) if f.endswith('.pth')]
        if pth_files:
            print(f"      .pth files: {', '.join(pth_files)}")
            for pth in pth_files:
                pth_path = os.path.join(sp, pth)
                try:
                    with open(pth_path, 'r') as f:
                        content = f.read().strip()
                        print(f"        {pth}: {content}")
                except:
                    pass

# 4. Config directory
print(f"\n{'Blender config directories:':-^80}")
possible_config_dirs = [
    os.path.expanduser(f"~/.config/blender/{blender_version}/python/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
    os.path.expanduser(f"~/snap/blender/common/.config/blender/{blender_version}/python/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
    os.path.expanduser("~/.config/blender/"),
    os.path.expanduser("~/snap/blender/"),
]

for config_dir in possible_config_dirs:
    exists = "✓" if os.path.exists(config_dir) else "✗"
    print(f"  {exists} {config_dir}")

# 5. Try to import utils
print(f"\n{'Import test:':-^80}")
try:
    import utils
    print(f"✓ Successfully imported utils from: {utils.__file__}")
except ImportError as e:
    print(f"✗ Failed to import utils: {e}")
    print(f"\n{'Suggested fix:':-^80}")

    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_gen_dir = os.path.join(os.path.dirname(current_dir), "image_generation")

    if not os.path.exists(image_gen_dir):
        image_gen_dir = os.getcwd()

    print(f"  Run this command to add image_generation to Python path:")
    print()

    for sp in site.getsitepackages():
        if os.path.exists(sp):
            pth_file = os.path.join(sp, "clevr.pth")
            print(f"  echo '{image_gen_dir}' >> {pth_file}")
            print(f"  # or")

    print()
    print(f"  Alternatively, add this at the start of your Python script:")
    print(f"  import sys")
    print(f"  sys.path.insert(0, '{image_gen_dir}')")

# 6. Check utils.py location
print(f"\n{'Looking for utils.py:':-^80}")
possible_locations = [
    os.path.join(os.getcwd(), "utils.py"),
    os.path.join(os.getcwd(), "image_generation", "utils.py"),
    os.path.expanduser("~/code/clevr-dataset-gen/image_generation/utils.py"),
]

for loc in possible_locations:
    exists = "✓" if os.path.exists(loc) else "✗"
    print(f"  {exists} {loc}")

print("\n" + "=" * 80)
