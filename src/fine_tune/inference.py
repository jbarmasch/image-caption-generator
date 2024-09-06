import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from config import CHECKPOINT_PATH, OUTPUT_PATH, TEST_IMAGES_PATH
from model import ImageCaptioningModel
from pathlib import Path

def generate_caption(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        caption = model.tokenizer.decode(output.argmax(dim=-1).cpu().numpy().flatten(), skip_special_tokens=True)
    
    return caption

def process_images(image_dir, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioningModel('resnet50', 'bert-base-uncased', 128).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    image_dir = Path(image_dir)
    output_dir = OUTPUT_PATH / "captions"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_dir.glob("*.jpg"):  # Adjust pattern if images are not .jpg
        image = Image.open(image_path).convert('RGB')
        caption = generate_caption(image, model, device)

        # Plotting image with caption
        plt.figure()
        plt.imshow(image)
        plt.title(caption)
        plt.axis('off')

        # Save the plot with caption
        output_image_path = output_dir / f"{image_path.stem}_captioned.jpg"
        plt.savefig(output_image_path, bbox_inches='tight')
        plt.close()
        print(f"Processed {image_path.name}: {caption}")

if __name__ == "__main__":
    img_dir = TEST_IMAGES_PATH
    checkpoint = CHECKPOINT_PATH / "model_epoch_10.pth"
    process_images(img_dir, checkpoint)