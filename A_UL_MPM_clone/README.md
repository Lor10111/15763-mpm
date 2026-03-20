# **A_UL_MPM_clone**

This folder contains a working implementation of the **Affine Unified Lagrangian Material Point Method (AULMPM)**, adapted from publicly available research code associated with the paper:

**“A Unified Lagrangian Material Point Method”**  
Source reference: https://orionquest.github.io/papers/AULMPM/paper.html

This clone preserves the original structure, helper modules, examples, and plugins, while organizing the codebase for reproducibility, compilation, and further development.

---

## **📌 Overview**

AULMPM is a variant of the Material Point Method (MPM) designed to improve stability, accuracy, and handling of complex deformation. This implementation includes:

- Core MPM data structures and solvers  
- Constitutive models  
- Collision helpers  
- Implicit and explicit force computation modules  
- Example scenes and geometry  
- TL‑APIC plugin implementations (2D and 3D)  
- Tools for grid traversal, influence iteration, and particle I/O  

The code is written primarily in C++ and organized into modular components for clarity and extensibility.

---

## **📁 Folder Structure**

### **Core Components**
- `MPM_Constitutive_Model.*` — constitutive model definitions  
- `MPM_Driver.*` — main simulation driver  
- `MPM_Example.*` — example simulation setups  
- `MPM_Object.h`, `MPM_Particle.h`, `MPM_Levelset.h` — core data structures  
- `Explicit_Force_Helper.h`, `Implicit_Force_Helper/` — force computation modules  

### **Plugins**
- `plugin_TL_APIC/` — TL‑APIC rendering and viewer integration  
- `tl_apic_2d/`, `tl_apic_3d/` — 2D and 3D APIC test implementations  

### **Examples**
- `examples/` — OBJ meshes, images, and sample scenes  
  - `bunny.obj`, `cow.obj`, `cheburashka.obj`, etc.  
  - `2D_bunny.txt`, `phi.txt`  
  - Useful for testing and benchmarking  

### **Tools**
- `Tools/` — matrix utilities, iterators, and helper classes  
- `Read_Write/` — particle I/O utilities  

### **Miscellaneous**
- `CImg.h` — header‑only image library  
- `image2txt.py` — image‑to‑text conversion script  
- `Log.md`, `Note.md` — developer notes  

---

## **🔧 Build Instructions**

This project uses **CMake**. Each module (e.g., TL‑APIC, 2D/3D tests) includes its own `CMakeLists.txt`.

Typical build steps:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Some modules may require additional dependencies (e.g., OpenMP, image libraries).

---

## **📦 Large Files & Git LFS**

This folder contains several large assets (OBJ meshes, `.txt` geometry files, images).  
To manage these efficiently, the repository uses **Git LFS**.

Tracked patterns include:

- `*.txt`
- `*.obj`
- `*.bin`
- `*.json`
- `A_UL_MPM_clone/*.h` (for consistency with original structure)

Before cloning or contributing, ensure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

---

## **📚 Original Source & Attribution**

This implementation is derived from publicly available research code accompanying the AULMPM paper:

**A Unified Lagrangian Material Point Method**  
Source reference: https://orionquest.github.io/papers/AULMPM/paper.html

This repository preserves the structure and functionality of the original code while enabling easier compilation, experimentation, and integration into larger MPM workflows.

---

## **🧭 Notes**

- This folder is intended as a self‑contained module within the broader `15763-mpm` repository.  
- Some components may rely on external libraries or specific compiler settings.  
- The codebase is suitable for research, experimentation, and educational use.  
