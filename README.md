#  Urban region representation learning with human trajectories: a multi-view approach incorporating transition, spatial, and temporal perspectives
  <img src="MTE.jpg">

## Requirements
  
- Python >= 3.8  
- torch = 1.13.0
- numpy = 1.25.2
- tqdm
- sklearn
- scipy
  

  
## Training  
  ```python
#train spatial view
python train.py --view_name='spatial'
# eval MTE
python tasks.py ----task='mte'
   ```
  
## Citation  
  If you find this repository, e.g., the paper, code, and the datasets, useful in your research, please cite the following paper:
