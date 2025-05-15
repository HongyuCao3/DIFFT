# DIFFT
This is the implementation code of DIFFT

## Implementation

### Step 1: construct the database
```
python3 utils/datacollection/GRFG_with_nni.py --file-name DATASETNAME 
```
### Step 2: VAE Training
```
python3 main.py ---task_name DATASETNAME 
```
### Step 2: VAE Training
```
python3 diffusion_main.py ---task_name DATASETNAME 
```

