import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import engine
import helper_functions

num_workers = os.cpu_count()  # number of cores on your machine
device = "cuda" if torch.cuda.is_available() else "cpu"  # Setup device-agnostic code

train_dir = "images/training"
test_dir = "images/testing"

# Write transform for image
data_transform = transforms.Compose(
    [
        # Resize the images to 224 x 224
        transforms.Resize(size=(224, 224)),
        # Flip the images randomly on the horizontal for data augmentation/ artificially change the data
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # p = probability of flip, 0.5 = 50% chance
        transforms.ToTensor(),  # Turn the image into a torch.Tensor and convert all pixel values from 0 to 255 to be between 0.0 and 1.0
    ]
)


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(
    root=train_dir,  # target folder of images
    transform=data_transform,  # transforms to perform on data (images)
    target_transform=None,  # transforms to perform on labels (if necessary)
)

test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)


# DataLoader objects to makes our dataset iterable so a model can go through learn the relationships between features and labels.
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=4,  # how many samples per batch?
    num_workers=num_workers,  # how many subprocesses to use for data loading? (higher = more)
    shuffle=True,
)  # shuffle the data at every epoch (iteration through the dataset)

test_dataloader = DataLoader(
    dataset=test_data, batch_size=1, num_workers=num_workers, shuffle=False
)  # don't usually need to shuffle testing data


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 20
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer for your classification task
    num_classes = len(train_data.classes)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=25088,
            out_features=num_classes,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(device)

    # Send the model to the target device
    model = model.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # for multi-class classification problems
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.00001, momentum=0.9)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Train model
    model_results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
    )
    # End the timer and print out how long it took
    end_time = timer()
    torch.save(model, "model.pth")
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    helper_functions.plot_loss_curves(model_results)
