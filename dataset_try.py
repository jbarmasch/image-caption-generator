from torch.utils.data import Dataset
from datasets import load_dataset

# Load dataset with just the "test" split
dataset = load_dataset('nlphuji/flickr30k', split='test', streaming=True)
# Create empty lists for train, validation, and test
train_filter = lambda x: x['split'] == 'train'
val_filter = lambda x: x['split'] == 'val'
test_filter = lambda x: x['split'] == 'test'

train_dataset = dataset.filter(train_filter)
val_dataset = dataset.filter(val_filter)
test_dataset = dataset.filter(test_filter)



# Print first ten entries for each dataset
i = 0
for entry in iter(train_dataset):
    if i < 10:
        print(f"{entry['filename']}: {entry['caption']} - {entry['split']}")
        # Show the image with its caption
        image = entry['image']
        print(f"Image shape: {type(image)}")
    else:
        break
    i += 1
print("\n")
i = 0
for entry in iter(val_dataset):
    if i < 10:
        print(f"{entry['filename']}: {entry['caption']} - {entry['split']}")
    else:
        break
    i += 1
print("\n")
i = 0
for entry in iter(test_dataset):
    if i < 10:
        print(f"{entry['filename']}: {entry['caption']} - {entry['split']}")
    else:
        break
    i += 1
