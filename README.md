# blech_clust

Python and R based code for clustering and sorting electrophysiology data
recorded using the Intan RHD2132 chips.  Originally written for cortical
multi-electrode recordings in Don Katz's lab at Brandeis.  Optimized for the
High performance computing cluster at Brandeis
(https://kb.brandeis.edu/display/SCI/High+Performance+Computing+Cluster) but
can be easily modified to work in any parallel environment. Visit the Katz lab
website at https://sites.google.com/a/brandeis.edu/katzlab/

### Order of operations  
1. `python blech_exp_info.py`  
    - Pre-clustering step. Annotate recorded channels and save experimental parameters  
    - Takes template for info and electrode layout as argument

2. `python blech_clust.py`
    - Setup directories and define clustering parameters  
3. `python blech_common_avg_reference.py`  
    - Perform common average referencing to remove large artifacts  
4. `bash blech_run_process.sh` 
    - Embarrasingly parallel spike extraction and clustering  

5. `python blech_post_process.py`  
    - Add selected units to HDF5 file for further processing  

6. `bash blech_run_QA.sh`  
    - Run quality asurance steps: 1) spike-time collisions across units, 2) drift within units
7. `python blech_units_plot.py`  
    - Plot waveforms of selected spikes  
8. `python blech_make_arrays.py`  
    - Generate spike-train arrays  
9. `python blech_make_psth.py`  
    - Plots PSTHs and rasters for all selected units  
10. `python blech_palatability_identity_setup.py`  
12. `python blech_overlay_psth.py`  
    - Plot overlayed PSTHs for units with respective waveforms  

### Setup
```
cd <path_to_blech_clust>/requirements                       # Move into blech_clust folder with requirements files
conda clean --all                                           # Removes unused packages and caches
conda create --name blech_clust python=3.8.13               # Create "blech_clust" environment with conda requirements
conda activate blech_clust                                  # Activate blech_clust environment
bash conda_requirements_base.sh                             # Install main packages using conda/mamba
bash install_gnu_parallel.sh                                # Install GNU Parallel
pip install -r pip_requirements_base.txt                    # Install pip requirements (not covered by conda)
bash patch_dependencies.sh                                  # Fix issues with dependencies
cd ~/Desktop                                                # Relocate to download classifier library
git clone https://github.com/abuzarmahmood/neuRecommend.git # Download classifier library
```
- Parameter files will need to be setup according to [Setting up params](https://github.com/abuzarmahmood/blech_clust/wiki/Getting-Started#setting-up-params)

### Convenience scripts
- blech_clust_pre.sh : Runs steps 2-5  
- blech_clust_post.sh : Runs steps 7-14   

### Operations Workflow Visual 
![blech_clust_readme](https://github.com/abuzarmahmood/blech_clust/assets/12436309/7ff483da-7aa6-4f96-af64-3fbe3205c6ef)



### Example workflow
```
DIR=/path/to/raw/data/files  
python blech_exp_info.py $DIR  # Generate metadata and electrode layout  
bash blech_clust_pre.sh $DIR   # Perform steps up to spike extraction and UMAP  
python blech_post_process.py   # Add sorted units to HDF5 (CLI or .CSV as input)  
bash blech_clust_post.sh       # Perform steps up to PSTH generation
```

### Test Dataset
We are grateful to Brandeis University Google Filestream for hosting this dataset <br>
Data to test workflow available at:<br>
https://drive.google.com/drive/folders/1ne5SNU3Vxf74tbbWvOYbYOE1mSBkJ3u3?usp=sharing
