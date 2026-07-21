
# Seminar-learning-from-demonstrations
### Trajectory Generation, GMM/GMR Composition for 2D Handwriting


## Project Overview

The primary goal of this project is to model human handwriting demonstrations, compose individual letters into full words (e.g., `"Basti"`) with pbdlib-matlab libary, and adapt movement primitives to new target positions in a stable and robust manner.

### Key Features:
**Data Preprocessing & Arc-Length Resampling:** 
   - Extraction of 2D trajectories from handmade SVG files and MATLAB `.mat` datasets (`B`, `a`, `s`, `t`, `i`).
   - Uniform arc-length re-parameterization to eliminate velocity artifacts.
**GMM/GMR Word Composition:**
   - Time-based initialization and Expectation-Maximization (EM) fitting.
**Dynamic Movement Primitives (DMP):**
   - Implementation of a non-linear spring-damper system governed by a canonical system ($s$).
   - Goal adaptation ($g_{\text{new}}$) and stabilization techniques to avoid numerical instabilities.

### Results
<img width="1247" height="1004" alt="untitled4" src="https://github.com/user-attachments/assets/9692ad83-8f9d-4986-9922-b86f80920e14" />
<img width="1247" height="1004" alt="untitled2" src="https://github.com/user-attachments/assets/803684b4-8975-42c2-8472-1311ea648c92" />
<img width="1247" height="999" alt="untitled3" src="https://github.com/user-attachments/assets/dcb5339c-3fe5-435a-9003-5ee85e07939e" />
<img width="840" height="713" alt="GMM_states_comparison" src="https://github.com/user-attachments/assets/0553b54f-851e-4686-a68d-581527cf5ee5" />

## Acknowledgements & References

* **PbDLib (Programming by Demonstration Library):** 
  This project utilizes data structures and concepts from the MATLAB version of **PbDLib**, developed by Dr. Sylvain Calinon (Idiap Research Institute). 
  - Repository: [pbdlib-matlab on GitLab](https://gitlab.idiap.ch/rli/pbdlib-matlab)
