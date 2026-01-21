"""
READY-TO-USE KAGGLE NOTEBOOK
============================
Copy ALL cells below into your Kaggle notebook in order.

Total: 8 cells
Time: ~3 hours with GPU
"""

# ============================================================================
# CELL 1: Environment Check
# ============================================================================
"""markdown
# 🏥 Enhanced CVFBJTL-BCD: Breast Cancer Diagnosis

**Target:** >98.18% accuracy (paper baseline)

**Setup:**
1. ✅ Enable GPU (Settings → Accelerator → GPU T4 x2)
2. ✅ Add BreaKHis dataset
3. ✅ Upload 4 Python files
4. ✅ Run all cells

**Time:** ~2-4 hours with GPU
"""

# ============================================================================
# CELL 2: Check Python & GPU
# ============================================================================
import sys
import os
import tensorflow as tf

print(f"Python: {sys.version[:6]}")
print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️  NO GPU! Enable: Settings → Accelerator → GPU")

# ============================================================================
# CELL 3: Install Packages
# ============================================================================
!pip install -q imbalanced-learn==0.11.0 albumentations==1.3.1
print("✅ Packages installed")

# ============================================================================
# CELL 4: List Input Datasets
# ============================================================================
print("📂 Available Datasets:")
if os.path.exists('/kaggle/input'):
    for item in os.listdir('/kaggle/input'):
        print(f"   - {item}")
else:
    print("   (Not on Kaggle)")

# ============================================================================
# CELL 5: Upload Python Files Setup
# ============================================================================
"""markdown
## 📤 Upload Required Files

**You need to upload 4 Python files:**
1. enhanced_cvfbjtl_bcd_model.py
2. breakhis_dataloader.py
3. advanced_explainability.py
4. kaggle_train_cvfbjtl_bcd.py

**How to upload:**
- Click "File" → "Add Utility Script" → Upload each .py file
- Files will be placed in /kaggle/working/

**Then run the cell below to verify.**
"""

# ============================================================================
# CELL 6: Verify Everything (AUTO-SETUP)
# ============================================================================
import os
import shutil
from pathlib import Path

print("="*60)
print("🔍 AUTOMATIC SETUP & VERIFICATION")
print("="*60)

# -----------------------------------------------
# 1. Find Dataset
# -----------------------------------------------
print("\n1. Finding BreaKHis dataset...")

dataset_paths = [
    '/kaggle/input/breakhis',
    '/kaggle/input/breakhis-dataset',
    '/kaggle/input/ambarish-breakhis'
]

dataset_found = None
for path in dataset_paths:
    if os.path.exists(path):
        # Find directory with benign/ and malignant/
        for root, dirs, files in os.walk(path):
            if 'benign' in dirs and 'malignant' in dirs:
                dataset_found = root
                break
        if dataset_found:
            break

if dataset_found:
    total_imgs = sum(1 for r, d, f in os.walk(dataset_found) 
                     for x in f if x.endswith('.png'))
    print(f"   ✅ Dataset: {dataset_found}")
    print(f"   📊 Images: {total_imgs}")
else:
    print("   ❌ Dataset NOT FOUND!")
    print("   Add dataset: https://www.kaggle.com/datasets/ambarish/breakhis")

# -----------------------------------------------
# 2. Check Python Files
# -----------------------------------------------
print("\n2. Checking Python files...")

required_files = [
    'enhanced_cvfbjtl_bcd_model.py',
    'breakhis_dataloader.py',
    'advanced_explainability.py',
    'kaggle_train_cvfbjtl_bcd.py',
    'kaggle_setup_helper.py'  # Optional
]

# Check in /kaggle/working
os.chdir('/kaggle/working')
files_present = []

for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
        files_present.append(file)
    else:
        # Try to find in input (if uploaded as dataset)
        for root, dirs, files_list in os.walk('/kaggle/input'):
            if file in files_list:
                src = os.path.join(root, file)
                shutil.copy2(src, f'/kaggle/working/{file}')
                print(f"   ✅ {file} (copied from input)")
                files_present.append(file)
                break
        else:
            if 'setup_helper' not in file:  # Optional file
                print(f"   ❌ {file} - MISSING!")

all_required = ['enhanced_cvfbjtl_bcd_model.py', 'breakhis_dataloader.py', 
                'advanced_explainability.py', 'kaggle_train_cvfbjtl_bcd.py']
all_present = all(f in files_present for f in all_required)

# -----------------------------------------------
# 3. Save Configuration
# -----------------------------------------------
if dataset_found and all_present:
    import json
    
    config = {
        'dataset_path': dataset_found,
        'working_dir': '/kaggle/working',
        'output_dir': '/kaggle/working/outputs'
    }
    
    with open('kaggle_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Configuration saved!")
    print(f"   Dataset: {dataset_found}")
    print(f"   Ready to train!")
else:
    print("\n❌ Setup incomplete:")
    if not dataset_found:
        print("   - Dataset missing")
    if not all_present:
        print("   - Python files missing")

print("="*60)

# ============================================================================
# CELL 7: Run Training
# ============================================================================
"""markdown
## 🚀 Training

**This will:**
1. Load BreaKHis dataset (200X magnification)
2. Apply Gabor filtering
3. Build Enhanced CVFBJTL-BCD model
4. Train for 50 epochs (~2-4 hours)
5. Generate evaluation plots
6. Save results to /kaggle/working/outputs/

**You can monitor progress below.**
"""

# ============================================================================
# CELL 8: Execute Training Script
# ============================================================================
import sys
sys.path.insert(0, '/kaggle/working')

# Run the complete training pipeline
%run kaggle_train_cvfbjtl_bcd.py

print("\n✅ Training complete!")
print("📁 Check /kaggle/working/outputs/ for results")

# ============================================================================
# CELL 9 (Optional): Download Results as ZIP
# ============================================================================
"""markdown
## 📥 Download Results

Results are saved in `/kaggle/working/outputs/`

**Option 1:** Download individual files (click folder icon)
**Option 2:** Create ZIP (run cell below)
"""

# ============================================================================
# CELL 9 (Optional): Download Results as ZIP
# ============================================================================
"""markdown
## 📥 Download Results

Results are saved in `/kaggle/working/outputs/`

**Option 1:** Download individual files (click folder icon)
**Option 2:** Create ZIP (run cell below)
"""

import shutil
import os

if os.path.exists('/kaggle/working/outputs'):
    # Create ZIP
    shutil.make_archive(
        '/kaggle/working/training_results',
        'zip',
        '/kaggle/working/outputs'
    )
    
    print("✅ Created: training_results.zip")
    print("\nContents:")
    for file in os.listdir('/kaggle/working/outputs'):
        size = os.path.getsize(f'/kaggle/working/outputs/{file}')
        print(f"   - {file} ({size/1024/1024:.1f} MB)")
    
    print("\n📥 To download:")
    print("   1. Click folder icon (left sidebar)")
    print("   2. Find: training_results.zip")
    print("   3. Click ... → Download")
else:
    print("❌ No outputs found. Training may not have completed.")
