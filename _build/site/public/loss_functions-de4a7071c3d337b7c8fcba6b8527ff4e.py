import torch

def mse(X_p, X_t):
    # Calculate the squared Euclidean distances between corresponding rows
    squared_distances = torch.sum((X_p - X_t) ** 2, dim = 1)

    # Calculate mean of the squared distances to get the loss
    loss = torch.mean(squared_distances)

    return loss


import torch.fft

def fourier_descriptor_matching_loss(X_p, X_t, num_descriptors = 20):
    # Compute Fourier transforms (using FFT)
    fft_p = torch.fft.fft(torch.complex(X_p[..., 0], X_p[..., 1]))
    fft_t = torch.fft.fft(torch.complex(X_t[..., 0], X_t[..., 1]))

    # Select relevant descriptors (low frequencies)
    F_p = fft_p[:num_descriptors]
    F_t = fft_t[:num_descriptors]

    # Calculate MSE loss on magnitudes or complex values
    loss = torch.mean(torch.abs(F_p - F_t)**2)
    return loss



def hausdorff_soft_loss(X_p, X_t, tau = 1.0):
    # Compute pairwise squared distances
    dist_matrix = torch.cdist(X_p, X_t)  # Shape: (N1, N2)
    
    # Compute soft-minimum distances
    d_soft_p1_to_p2 = tau * torch.logsumexp(dist_matrix / tau, dim = 1)  # Shape: (N1,)
    d_soft_p2_to_p1 = tau * torch.logsumexp(dist_matrix.T / tau, dim = 1)  # Shape: (N2,)

    # Compute the soft Hausdorff loss
    hausdorff_soft = d_soft_p1_to_p2.mean() + d_soft_p2_to_p1.mean()
    
    return hausdorff_soft



# def mse_exp_loss(X_p, X_t):
#     # Calculate the squared Euclidean distances between corresponding rows
#     squared_distances = torch.sum((X_p - X_t) ** 2, dim = 1)

#     # Calculate mean of the squared distances to get the loss
#     tau = 0.01
#     print(tau)

#     loss = tau * torch.log(torch.mean(torch.exp(squared_distances / tau)))

#     return loss