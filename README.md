# ski-qos-jaamas-2023
Experiments for jaamas paper.

### 1. Download datasets
Execute the command ```python -m setup load_datasets [-f] [-o]``` to download datasets from UCI website.
By default, the command will store the original dataset into ```datasets``` folder.
If you specify:
- ```-f y``` to binarize input features;
- ```-o y``` to map the output classes into numeric indices.

Datasets are not tracked by git, so you first need to execute this command before doing anything else.