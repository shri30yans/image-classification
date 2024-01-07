import helper_functions
from pathlib import Path
import torch
from torchvision import datasets

image_path = "images\\training"
data_set = datasets.ImageFolder(root=image_path)
class_names = data_set.classes

num_images_to_plot = 2
test_image_path_list = list(
    Path(image_path).glob("*/*")
)  # get list all image paths from test data

model = torch.load("model.pth")

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# Make predictions on and plot the images
correct = 0
for image_path in test_image_path_list:
    helper_functions.pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=class_names,
                        # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                        image_size=(224, 224))

    # correct_result = helper_functions.pred_image(
    #     model=model,
    #     image_path=image_path,
    #     class_names=class_names,
    #     # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
    #     image_size=(224, 224),
    # )

    # if correct_result:
    #     correct += 1

print(f"Accuracy: {correct/len(test_image_path_list)}")
