### Using PACS data for domain label shuffling


## Data
### PACDataset
PACS data is located in `../data/pacs` (or name yourself). 

```{python}
from PAC_Dataset import PACDataset
p_dataset_train = PACDataset('p', split='train')
p_dataset_test = PACDataset('p', split='test')
```

The cached `.npy` file for all data needed is located at `../data/pacs/cache`, i.e. the same root folder of your PACS data folder under name `cache`.

Once generated, the train and test data will be stored until you delete the `cache` folder. 

If you want to replace the old version of concatenated version of `.npy` file, you can do as the following:

```{python}
from PAC_Dataset import PACDataset, ConcatDataset
p_dataset_train = PACDataset('p', split='train', reload=True)
p_dataset_test = PACDataset('p', split='test', reload=True)
```

__Note__: default of `split` argument has been set to `train`.



### ConcatDataset
ConcatDataset is used for concat all instance of `PAC_Dataset` dataset together.

```
from PAC_Dataset import PACDataset, ConcatDataset
p_dataset_train = PACDataset('p', split='train', reload=True)
p_dataset_test = PACDataset('p', split='test', reload=True)
concat_dataset = ConcatDataset(p_dataset_train, p_dataset_test)
```
