from fine_tune import collate_fn
from dataset import CustomDataset

dataset = CustomDataset()
train_dataset, val_dataset, test_dataset = dataset.get_datasets()

batch_list = [train_dataset[i] for i in range(5)]

images = [i for i, t, l, a in batch_list]

print(f"Images beginning: {images}")

collate_fn(batch_list)
