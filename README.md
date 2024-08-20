#  Urban region representation learning with human trajectories: a multi-view approach incorporating transition, spatial, and temporal perspectives
This project is linked to a paper published in GISCIENCE & Remote Sensing: Urban region representation learning with human trajectories: a multi-view approach incorporating transition, spatial, and temporal perspectives. In this study, we propose a novel approach MTE for learning effective region representations with human trajectories in a fully unsupervised manner. MTE models three salient information perspectives of trajectory data, transition, spatial, and temporal views, and utilizes varying machine learning techniques based on the traits of different views. 
  <img src="MTE.jpg">

## Requirements
  
- Python >= 3.8  
- torch = 1.13.0
- numpy = 1.25.2
- tqdm
- sklearn
- scipy

## Structure
There are two main scripts in this repository: train.py and tasks.py.

### train.py
train.py is used to learn the region representations. Due to data privacy protection restrictions, we cannot provide the original data. However, we have manually created some training data to help readers understand our code. The pre-trained base station transition embeddings are loaded, and graph contrastive learning techniques are used to separately learn the embeddings for the spatial and temporal views, which are then mapped into regional representations using Voronoi polygons.
  ```python
#train spatial view
python train.py --view_name='spatial'
```

### tasks.py
The MTE embeddings and their variants, MTE-temporal and MTE-spatial embeddings, are loaded and evaluated on land use classification, population density estimation, and housing price prediction.
  ```python
# eval MTE
python tasks.py --task='mte'
   ```
  
## Citation  
  If you find this repository, e.g., the paper, code, and the datasets, useful in your research, please cite the following paper:
