from dataset import CustomDataset

dataset = CustomDataset()
train_dataset, val_dataset, test_dataset = dataset.get_datasets()

i = 0
j= 0
k=0
print("Counting datasets")
print()
print("Counting train dataset...")
for batch in train_dataset:
    i += 1
print(f"Train dataset count: {i}")
print()
print("Counting validation dataset...")
for batch in val_dataset:
    j += 1
print(f"Validation dataset count: {j}")
print()
print("Counting test dataset...")
for batch in test_dataset:
    k += 1
print(f"Test dataset count: {k}")
