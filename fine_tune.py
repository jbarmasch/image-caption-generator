from models.moondream import MoondreamCaptioner
import torch

def main():
    dataset_name = 'flickr30k'
    output_dir = './Training results/Weights/Moondream'
    num_train_epochs = 5
    batch_size = 4
    save_steps = 500

    captioner = MoondreamCaptioner(torch_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    captioner.fine_tune(
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        save_steps=save_steps
    )

if __name__ == "__main__":
    main()
