#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAGGLE SETUP SCRIPT
===================
Auto-configuration for Kaggle environment

This script:
1. Detects all necessary files
2. Finds BreaKHis dataset automatically
3. Sets up paths correctly
4. Verifies all dependencies
5. Provides clear error messages

Usage in Kaggle:
    %run kaggle_setup_helper.py
"""

import os
import sys
import subprocess
from pathlib import Path
import json


class KaggleSetup:
    """Automatic Kaggle environment setup"""
    
    def __init__(self):
        self.is_kaggle = os.path.exists('/kaggle')
        self.working_dir = '/kaggle/working' if self.is_kaggle else '.'
        self.input_dir = '/kaggle/input' if self.is_kaggle else '.'
        self.dataset_path = None
        self.errors = []
        self.warnings = []
        
    def print_header(self):
        """Print setup header"""
        print("\n" + "="*70)
        print("🔧 KAGGLE ENVIRONMENT SETUP")
        print("="*70)
        print(f"Running on Kaggle: {self.is_kaggle}")
        print(f"Working Directory: {self.working_dir}")
        print(f"Input Directory: {self.input_dir}")
        print("="*70 + "\n")
    
    def check_gpu(self):
        """Check GPU availability"""
        print("1️⃣  Checking GPU...")
        print("-" * 50)
        
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                print(f"✅ GPU Available: {len(gpus)} device(s)")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Memory growth enabled")
                
                return True
            else:
                self.warnings.append("No GPU detected")
                print("⚠️  No GPU found!")
                print("   Go to: Settings → Accelerator → GPU T4 x2")
                return False
                
        except Exception as e:
            self.errors.append(f"GPU check failed: {e}")
            print(f"❌ Error checking GPU: {e}")
            return False
    
    def find_dataset(self):
        """Find BreaKHis dataset in Kaggle input"""
        print("\n2️⃣  Finding BreaKHis dataset...")
        print("-" * 50)
        
        if not self.is_kaggle:
            # Local mode
            local_paths = [
                'BreaKHis_v1/histology_slides/breast',
                '../BreaKHis_v1/histology_slides/breast',
                'dataset/breast'
            ]
            
            for path in local_paths:
                if os.path.exists(path):
                    self.dataset_path = path
                    print(f"✅ Found dataset: {path}")
                    return True
            
            self.errors.append("BreaKHis dataset not found locally")
            print("❌ Dataset not found in local directories")
            return False
        
        # Kaggle mode - search in /kaggle/input
        possible_names = [
            'breakhis',
            'breakhis-dataset', 
            'ambarish-breakhis',
            'breast-cancer-histopathological'
        ]
        
        print(f"Searching in: {self.input_dir}")
        
        # List all input datasets
        if os.path.exists(self.input_dir):
            datasets = os.listdir(self.input_dir)
            print(f"Found {len(datasets)} dataset(s):")
            for ds in datasets:
                print(f"   - {ds}")
        
        # Try to find BreaKHis dataset
        for name in possible_names:
            path = os.path.join(self.input_dir, name)
            if os.path.exists(path):
                # Find directory with benign/ and malignant/ folders
                for root, dirs, files in os.walk(path):
                    if 'benign' in dirs and 'malignant' in dirs:
                        self.dataset_path = root
                        
                        # Count images
                        total_images = 0
                        for r, d, f in os.walk(self.dataset_path):
                            total_images += len([x for x in f if x.endswith('.png')])
                        
                        print(f"\n✅ Dataset found: {self.dataset_path}")
                        print(f"   Total images: {total_images}")
                        
                        if total_images < 100:
                            self.warnings.append(f"Only {total_images} images found - expected ~7,900")
                        
                        return True
        
        self.errors.append("BreaKHis dataset not found in Kaggle input")
        print("\n❌ BreaKHis dataset NOT found!")
        print("\n📥 To add dataset to Kaggle:")
        print("   1. Go to: https://www.kaggle.com/datasets/ambarish/breakhis")
        print("   2. Click 'New Notebook' (adds dataset automatically)")
        print("   OR")
        print("   1. In this notebook: Click '+ Add Data'")
        print("   2. Search: 'ambarish breakhis'")
        print("   3. Click 'Add'")
        print("   4. Restart kernel")
        
        return False
    
    def check_python_files(self):
        """Check if all required Python files exist"""
        print("\n3️⃣  Checking Python files...")
        print("-" * 50)
        
        required_files = [
            'enhanced_cvfbjtl_bcd_model.py',
            'breakhis_dataloader.py',
            'advanced_explainability.py',
            'kaggle_train_cvfbjtl_bcd.py'
        ]
        
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(self.working_dir, file)
            
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {file} ({size:,} bytes)")
            else:
                missing_files.append(file)
                print(f"❌ {file} - NOT FOUND")
        
        if missing_files:
            self.errors.append(f"Missing files: {', '.join(missing_files)}")
            print(f"\n❌ Missing {len(missing_files)} file(s)")
            print("\n📤 To upload files to Kaggle:")
            print("   Method 1 (Recommended):")
            print("      1. Click 'File' → 'Add Utility Script'")
            print("      2. Upload each .py file")
            print("   Method 2:")
            print("      1. Create dataset with Python files")
            print("      2. Add dataset to notebook")
            print("      3. Copy files to /kaggle/working/")
            return False
        
        print(f"\n✅ All {len(required_files)} Python files present!")
        return True
    
    def check_dependencies(self):
        """Check required packages"""
        print("\n4️⃣  Checking dependencies...")
        print("-" * 50)
        
        required_packages = {
            'tensorflow': 'tensorflow',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'imblearn': 'imbalanced-learn',
            'albumentations': 'albumentations'
        }
        
        missing_packages = []
        
        for module_name, package_name in required_packages.items():
            try:
                __import__(module_name)
                print(f"✅ {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                print(f"❌ {package_name} - NOT INSTALLED")
        
        if missing_packages:
            print(f"\n📦 Installing missing packages...")
            for package in missing_packages:
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', '-q', package
                    ])
                    print(f"   ✅ Installed: {package}")
                except Exception as e:
                    self.errors.append(f"Failed to install {package}: {e}")
                    print(f"   ❌ Failed: {package}")
        else:
            print("\n✅ All dependencies installed!")
        
        return len(missing_packages) == 0
    
    def save_config(self):
        """Save configuration for training script"""
        print("\n5️⃣  Saving configuration...")
        print("-" * 50)
        
        config = {
            'dataset_path': self.dataset_path,
            'working_dir': self.working_dir,
            'is_kaggle': self.is_kaggle
        }
        
        config_path = os.path.join(self.working_dir, 'kaggle_config.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Configuration saved to: {config_path}")
            print(f"   Dataset path: {self.dataset_path}")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to save config: {e}")
            print(f"❌ Failed to save configuration: {e}")
            return False
    
    def print_summary(self):
        """Print setup summary"""
        print("\n" + "="*70)
        print("📊 SETUP SUMMARY")
        print("="*70)
        
        if self.errors:
            print(f"\n❌ {len(self.errors)} ERROR(S):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNING(S):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if not self.errors:
            print("\n✅ SETUP COMPLETE!")
            print("\n🚀 Next steps:")
            print("   1. Run training script:")
            print("      %run kaggle_train_cvfbjtl_bcd.py")
            print("\n   2. Or use the main notebook cells")
            print("\n⏱️  Expected training time:")
            print("   - With GPU: 2-4 hours")
            print("   - Without GPU: 24+ hours")
        else:
            print("\n❌ SETUP INCOMPLETE")
            print("   Please fix the errors above before training.")
        
        print("="*70 + "\n")
        
        return len(self.errors) == 0
    
    def run(self):
        """Run complete setup"""
        self.print_header()
        
        # Run all checks
        gpu_ok = self.check_gpu()
        dataset_ok = self.find_dataset()
        files_ok = self.check_python_files()
        deps_ok = self.check_dependencies()
        
        # Save configuration if everything is OK
        if dataset_ok:
            config_ok = self.save_config()
        else:
            config_ok = False
        
        # Print summary
        success = self.print_summary()
        
        return success


# Run setup when script is executed
if __name__ == "__main__":
    setup = KaggleSetup()
    success = setup.run()
    
    if success:
        print("✅ Ready to train!")
    else:
        print("⚠️  Please fix setup issues before training.")
