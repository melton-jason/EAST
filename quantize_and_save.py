import torch
from torch.utils.data import DataLoader
from dataset import custom_dataset
from model import EAST
import os

torch.backends.quantized.engine = 'qnnpack'

def calibrate_with_data(model, img_path, gt_path, batch_size=8, num_batches=10):
    dataset = custom_dataset(img_path, gt_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.eval()
    with torch.no_grad():
        for i, (img, _, _, _) in enumerate(loader):
            model(img)
            if i >= num_batches:
                break


if __name__ == "__main__":
    # Load float model
    model = EAST(pretrained=True)
    model.eval()
    model.load_state_dict(torch.load("pths/east_vgg16.pth", map_location="cpu"))

    # Prepare for quantization
    model.prepare_for_quantization()

    # Calibrate on training data
    train_img_path = os.path.abspath("../ICDAR_2015/train_img")
    train_gt_path  = os.path.abspath("../ICDAR_2015/train_gt")
    calibrate_with_data(model, train_img_path, train_gt_path)

    # Convert to quantized
    quantized_model = torch.quantization.convert(model.eval(), inplace=False)

    # Save
    os.makedirs("pths_quantized", exist_ok=True)
    torch.save(quantized_model.state_dict(), "pths_quantized/east_quantized.pth")

    print("✅ Quantized model saved to pths_quantized/east_quantized.pth")
