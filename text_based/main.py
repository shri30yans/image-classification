import pandas
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

def data_preprocessing(dataframe):

    """
    Purpose:
    ---
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that
    there are features in the csv file whose values are textual (eg: Industry,
    Education Level etc)These features might be required for training the model
    but can not be given directly as strings for training. Hence this function
    should return encoded dataframe in which all the textual features are
    numerically labeled.

    Input Arguments:
    ---
    `dataframe`: [Dataframe]
                                              Pandas dataframe read from the provided dataset

    Returns:
    ---
    `encoded_dataframe` : [ Dataframe ]
                                              Pandas dataframe that has all the features mapped to
                                              numbers starting from zero

    Example call:
    ---
    encoded_dataframe = data_preprocessing(dataframe)
    """
    encoder = LabelEncoder()

    # Encode the categorical features
    dataframe["Education"] = encoder.fit_transform(
        dataframe["Education"]
    )
    dataframe["City"] = encoder.fit_transform(dataframe["City"])
    dataframe["Gender"] = encoder.fit_transform(dataframe["Gender"])
    dataframe["EverBenched"] = encoder.fit_transform(
        dataframe["EverBenched"]
    )

    scaler = StandardScaler()
    # Scale the numerical features
    # StandardScaler() helps to get standardized distribution, with a zero mean and standard deviation of one (unit variance).
    # It standardizes features by subtracting the mean value from the feature and then dividing the result by feature standard deviation.
    numerical_features = [
        "JoiningYear",
        "PaymentTier",
        "Age",
        "ExperienceInCurrentDomain",
    ]

    dataframe[numerical_features] = scaler.fit_transform(
        dataframe[numerical_features]
    )

    encoded_dataframe = dataframe
    return encoded_dataframe


def identify_features_and_targets(encoded_dataframe):
    """
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second
    item is the target label

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                                            Pandas dataframe that has all the features mapped to
                                            numbers starting from zero

    Returns:
    ---
    `features_and_targets` : [ list ]
                                                    python list in which the first item is the
                                                    selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    """
    # Define the features (all columns except 'LeaveOrNot')
    features = encoded_dataframe.drop(columns=["LeaveOrNot"])

    # Define the target variable ('LeaveOrNot')
    target = encoded_dataframe["LeaveOrNot"]
    features_and_targets = [features, target]
    return features_and_targets


# Function to load data as tensors
def load_as_tensors(features_and_targets):

    """
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training
    and validation, and then load them as as tensors.
    Training of the model requires iterating over the training tensors.
    Hence the training sensors need to be converted to iterable dataset
    object.

    Input Arguments:
    ---
    `features_and targets` : [ list ]
                                                    python list in which the first item is the
                                                    selected features and second item is the target label

    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
        Items:
        [0]: X_train_tensor: Training features loaded into Pytorch array
        [1]: X_test_tensor: Feature tensors in validation data
        [2]: y_train_tensor: Training labels as Pytorch tensor
        [3]: y_test_tensor: Target labels as tensor in validation data
        [4]: Iterable dataset object and iterating over it in
                    batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    """

    # Split data into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(features_and_targets[0]))
    X_train = features_and_targets[0][:train_size]
    y_train = features_and_targets[1][:train_size]
    X_val = features_and_targets[0][train_size:]
    y_val = features_and_targets[1][train_size:]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create iterable dataset objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    tensors_and_iterable_training_data = [
        X_train_tensor,
        X_val_tensor,
        y_train_tensor,
        y_val_tensor,
        train_dataset,
        val_dataset,
    ]
    return tensors_and_iterable_training_data


# Define the neural network model
class Salary_Predictor(nn.Module):

    """
    Purpose:
    ---
    The architecture and behavior of your neural network model will be
    defined within this class that inherits from nn.Module. Here you
    also need to specify how the input data is processed through the layers.
    It defines the sequence of operations that transform the input data into
    the predicted output. When an instance of this class is created and data
    is passed through it, the `forward` method is automatically called, and
    the output is the prediction of the model based on the input data.

    Returns:
    ---
    `predicted_output` : Predicted output for the given input data
    """

    def __init__(self):
        super(Salary_Predictor, self).__init__()
        """
		Type and number of layers
		"""
        self.number_of_features = 8
        self.fc1 = nn.Linear(self.number_of_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Activation functions
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        predicted_output = x

        return predicted_output


# Function to define the loss function
def model_loss_function():

    """
    Purpose:
    ---
    To define the loss function for the model. Loss function measures
    how well the predictions of a model match the actual target values
    in training data.

    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    """
    loss_function = nn.BCELoss()
    return loss_function  # Binary Cross-Entropy Loss for binary classification


# Function to define the optimizer
def model_optimizer(model):

    """
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible
    foating the parameters (weights and biases) in a way that
    minimizes the loss function.

    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class

    Returns:
    ---
    `optimizer`: Pre-defined optimizer from Pytorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Using Adam optimizer
    return optimizer


# Function to specify the number of training epochs
def model_number_of_epochs():
    """
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    """
    number_of_epochs = 100
    return number_of_epochs


# Function for training the model
def training_function(
    model,
    number_of_epochs,
    tensors_and_iterable_training_data,
    loss_function,
    optimizer,
):
    """
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors
                                                                                     and iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    """
    (
        X_train_tensor,
        _,
        y_train_tensor,
        _,
        train_dataset,
        _,
    ) = tensors_and_iterable_training_data

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(number_of_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()

    trained_model = model
    return trained_model


# Function for model validation and accuracy calculation
def validation_function(trained_model, tensors_and_iterable_training_data):

    """
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors
        and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    """
    (
        _,
        X_val_tensor,
        _,
        y_val_tensor,
        _,
        val_dataset,
    ) = tensors_and_iterable_training_data
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = trained_model(batch_X)
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y.view(-1, 1)).sum().item()

    model_accuracy = 100 * (correct / total)
    return model_accuracy


if __name__ == "__main__":

    # reading the provided dataset csv file using pandas library and
    # converting it to a pandas Dataframe
    dataframe = pandas.read_csv("dataset.csv")

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # training the model
    trained_model = training_function(
        model,
        number_of_epochs,
        tensors_and_iterable_training_data,
        loss_function,
        optimizer,
    )

    # validating and obtaining accuracy
    model_accuracy = validation_function(
        trained_model, tensors_and_iterable_training_data
    )
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(
        torch.jit.trace(model, (x)), "trained_model.pth"
    )
