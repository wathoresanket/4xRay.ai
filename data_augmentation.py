import os
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from models.unet.utils import load_seunet
import torch

class SegmentationPredictor:
    def __init__(self, model, mean=0.458, std=0.173, cuda=True) -> None:
        self.model = model
        self.model.eval()
        self.cuda = cuda
        self.mean = mean
        self.std = std

    def predict(self, image: np.ndarray) -> torch.Tensor:
        origin_shape = image.shape
        image = Image.fromarray(image).resize((256, 256))
        image = F.to_tensor(image)
        image = F.normalize(image, [self.mean], [self.std])
        if self.cuda:
            image = image.cuda()
        image = image.unsqueeze(0)

        predictions = self.model(image)
        predictions = predictions.squeeze(0)
        predictions = torch.argmax(predictions, dim=0, keepdim=True)
        predictions = F.resize(predictions, origin_shape, F.InterpolationMode.NEAREST)
        return predictions.squeeze(0)

def data_augmentation_left(image, mask, model, cuda=True):
    # Convert PIL image to numpy array
    image = np.array(image.convert('L'))
    mask = np.array(mask.convert('L'))

    # Create a predictor
    predictor = SegmentationPredictor(model, cuda=cuda)

    # Predict the image
    prediction = predictor.predict(image)

    # Only show quadrants 0 and 3
    prediction[(prediction != 1) & (prediction != 4)] = 0

    # Convert prediction to numpy for masking
    prediction_np = prediction.cpu().numpy()

    # Create a mask where prediction is greater than 0
    binary_mask = prediction_np > 0

    # Apply the mask to the original image
    image = image * binary_mask

    # Find the rightmost point in the predicted region
    y, x = np.where(binary_mask)
    rightmost_point = max(x)

    # Cut the right side of the image from the rightmost point
    image = image[:, :rightmost_point]

    # Flip the image along y-axis and create a copy
    flipped_image = np.fliplr(image.copy())

    # Join the original and flipped images horizontally
    new_image = np.hstack((image, flipped_image))

    # Apply the mask to the original image
    mask = mask * binary_mask

    # Cut the right side of the mask from the rightmost point obtained earlier
    mask = mask[:, :rightmost_point]

    # Get the shape of the mask
    height, width = mask.shape

    # Iterate over each pixel in the mask
    for y in range(height):
        for x in range(width):
            # If the grayscale value of the pixel is not between 1-8 or 25-32, make it total black (0)
            if not (1 <= mask[y, x] <= 8 or 25 <= mask[y, x] <= 32):
                mask[y, x] = 0

    # Flip the mask along y-axis and create a copy
    flipped_mask = np.fliplr(mask.copy())

    # Iterate over each pixel in the mask
    for y in range(height):
        for x in range(width):
            # If the grayscale value of the pixel is between 1-8, add 8 to it
            if 1 <= flipped_mask[y, x] <= 8:
                flipped_mask[y, x] += 8
            # If the grayscale value of the pixel is between 25-32, subtract 8 from it
            elif 25 <= flipped_mask[y, x] <= 32:
                flipped_mask[y, x] -= 8

    # Join the original and flipped masks horizontally
    new_mask = np.hstack((mask, flipped_mask))

    return Image.fromarray(new_image), Image.fromarray(new_mask)


def data_augmentation_right(image, mask, model, cuda=True):
    # Convert PIL image to numpy array
    image = np.array(image.convert('L'))
    mask = np.array(mask.convert('L'))

    # Create a predictor
    predictor = SegmentationPredictor(model, cuda=cuda)

    # Predict the image
    prediction = predictor.predict(image)

    # Only show quadrants 1 and 2
    prediction[(prediction != 2) & (prediction != 3)] = 0 

    # Convert prediction to numpy for masking
    prediction_np = prediction.cpu().numpy()

    # Create a mask where prediction is greater than 0
    binary_mask = prediction_np > 0

    # Apply the mask to the original image
    image = image * binary_mask

    # Find the leftmost point in the predicted part
    y, x = np.where(prediction.cpu().numpy() > 0)
    leftmost_point = min(x)

    # Cut the left side of the image from the leftmost point
    image = image[:, leftmost_point:]

    # Flip the image along y-axis and create a copy
    flipped_image = np.fliplr(image.copy())

    # Join the original and flipped images horizontally
    new_image = np.hstack((flipped_image, image))

    # Apply the mask to the original image
    mask = mask * binary_mask

    # Cut the left side of the mask from the leftmost point
    mask = mask[:, leftmost_point:]

    # Get the shape of the mask
    height, width = mask.shape

    # Iterate over each pixel in the mask
    for y in range(height):
        for x in range(width):
            # If the grayscale value of the pixel is not between 1-8 or 25-32, make it total black (0)
            if not (9 <= mask[y, x] <= 16 or 17 <= mask[y, x] <= 24):
                mask[y, x] = 0

    # Flip the mask along y-axis and create a copy
    flipped_mask = np.fliplr(mask.copy())

    # Iterate over each pixel in the mask
    for y in range(height):
        for x in range(width):
            # If the grayscale value of the pixel is between 1-8, add 8 to it
            if 9 <= flipped_mask[y, x] <= 16:
                flipped_mask[y, x] -= 8
            # If the grayscale value of the pixel is between 25-32, subtract 8 from it
            elif 17 <= flipped_mask[y, x] <= 24:
                flipped_mask[y, x] += 8

    # Join the original and flipped masks horizontally
    new_mask = np.hstack((flipped_mask, mask))

    return Image.fromarray(new_image), Image.fromarray(new_mask)


def data_augmentation_flip(image, mask):
    # Convert PIL image to numpy array
    image = np.array(image.convert('L'))
    mask = np.array(mask.convert('L'))

    # Flip the image along y-axis and create a copy
    new_image = np.fliplr(image.copy())

    # Flip the mask along y-axis and create a copy
    new_mask = np.fliplr(mask.copy())

    # Iterate over each pixel in the mask
    height, width = new_mask.shape
    for y in range(height):
        for x in range(width):
            # If the grayscale value of the pixel is between 1-8, add 8 to it
            if 9 <= new_mask[y, x] <= 16:
                new_mask[y, x] -= 8
            # If the grayscale value of the pixel is between 25-32, subtract 8 from it
            elif 17 <= new_mask[y, x] <= 24:
                new_mask[y, x] += 8
            # If the grayscale value of the pixel is between 1-8, add 8 to it
            elif 1 <= new_mask[y, x] <= 8:
                new_mask[y, x] += 8
            # If the grayscale value of the pixel is between 25-32, subtract 8 from it
            elif 25 <= new_mask[y, x] <= 32:
                new_mask[y, x] -= 8

    return Image.fromarray(new_image), Image.fromarray(new_mask)
