import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def automate_training(
        model,
        loss_fn,
        X_train,
        Y_train,
        epochs = 1000,
        print_cost_every = 200,
        learning_rate = 0.001,
):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

    num_digits = len(str(epochs))

    for epoch in range(epochs):
        Y_model = model(X_train)
        loss = loss_fn(Y_model, Y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss.item())

        if epoch == 0 or (epoch + 1) % print_cost_every == 0:
            print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')



def plot_curves(X_p, X_t):
    X_p = X_p.detach().numpy()
    plt.fill(X_t[:, 0], X_t[:, 1], color = "#C9C9F5", alpha = 0.46, label = "Target Curve")
    plt.fill(X_p[:, 0], X_p[:, 1], color = "#F69E5E", alpha = 0.36, label = "Fitted Curve")

    plt.plot(X_t[:, 0], X_t[:, 1], color = "#000000", linewidth = 2)
    plt.plot(X_p[:, 0], X_p[:, 1], color = "#000000", linewidth = 2, linestyle = "--")

    plt.axis('equal')
    plt.show()