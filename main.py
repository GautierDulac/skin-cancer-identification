###Imports
from preprocessing import split_train_valid_sets


###Constants
data_dir = 'data'
batch_size_train=64
batch_size_val=5
shuffle_train=True
shuffle_val=False
num_workers=6

###Main function
loader_train, loader_vali = split_train_valid_sets(data_dir, batch_size_train, batch_size_val, shuffle_train, shuffle_val, num_workers)