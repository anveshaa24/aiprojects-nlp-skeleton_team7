import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01) # use L2 regularization to prevent overfitting and simplify model, lambda = 0.01
    loss_fn = nn.BCELoss() #use when there is only one node in the output layer that stores a value from 0 to 1 where 0 and 1 are different classes

    max_accuracy = 0.0
    train_losses = []
    step = 1
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        losses = []
        total = 0

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Forward propagate
            inputs, targets = batch
            model.zero_grad()
            outputs = model(inputs)
            # TODO: Backpropagation and gradient descent
            loss = loss_fn(outputs.squeeze(1), targets.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total += 1

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                test_accuracy = evaluate(val_loader, model, loss_fn)
                print("Test accuracy: " + str(test_accuracy))
                if(test_accuracy > max_accuracy):
                    max_accuracy = test_accuracy
                    torch.save(model, "Bag_of_Words_Dense_Model.pt")
                model.train()
                #print(loss.item())

            step += 1
        epoch_loss = sum(losses) / total
        train_losses.append(epoch_loss)
        print(epoch_loss)


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """


    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    total_batches = 0.0
    total_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(inputs)
            total_accuracy += compute_accuracy(outputs.squeeze(1), labels)
            total_batches += 1
    return total_accuracy/total_batches
