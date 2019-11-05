###Imports
from preprocessing import split_train_valid_sets
from modelling import preconvfeat


###Constants
data_dir = 'data'
batch_size_train=64
batch_size_val=5
shuffle_train=True
shuffle_val=False
num_workers=6


###Main function
dset_sizes, loader_train, loader_valid = split_train_valid_sets(data_dir, batch_size_train, batch_size_val, shuffle_train, shuffle_val, num_workers)


