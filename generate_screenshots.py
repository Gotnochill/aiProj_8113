#!/usr/bin/env python3
"""
Generate screenshots/visualizations for the project report.

USAGE:
    python3 generate_screenshots.py

This script collects all PNG visualizations from part_X/out/ directories
and copies them to the screenshots/ folder for easy access.

TO GENERATE NEW VISUALIZATIONS:
    cd part_1 && python3 demo_visualize_multi_head.py && cd ..
    cd part_2 && python3 orchestrator.py && cd ..
    cd part_5 && python3 demo_moe.py && cd ..
    
Then run this script to collect them all.
"""

import os
import sys
import shutil
from pathlib import Path

def collect_existing_images():
    """Collect all existing images from part_X/out/ directories."""
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    print("Collecting existing visualizations...")
    count = 0
    
    for i in range(1, 10):
        out_dir = Path(f"part_{i}/out")
        if out_dir.exists():
            for img in out_dir.glob("*.png"):
                dest = screenshots_dir / f"{i:02d}_{img.name}"
                shutil.copy2(img, dest)
                print(f"  ✓ {img.relative_to('.')}")
                count += 1
    
    return count

def main():
    print("=" * 60)
    print("SCREENSHOT COLLECTION FOR LLM FROM SCRATCH")
    print("=" * 60)
    print()
    
    # Collect existing images
    count = collect_existing_images()
    
    print()
    print("=" * 60)
    if count > 0:
        print(f"✓ Collected {count} visualization(s) to screenshots/")
        print("\nView them:")
        print("  ls screenshots/")
    else:
        print("No visualizations found.")
        print("\nTo generate visualizations, run demos in each part:")
        print("  cd part_1 && python3 demo_visualize_multi_head.py")
        print("  cd part_2 && python3 orchestrator.py")
        print("  cd part_5 && python3 demo_moe.py")
        print("\nThen run this script again to collect them.")
    print("=" * 60)

if __name__ == "__main__":
    main()
