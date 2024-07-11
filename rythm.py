import torch
import numpy as np


def calculate_Tg_numpy(x, t_values, d_values, a_m):
    """
    Calculates the Tg matrix for given input data and parameters using NumPy.
    """

    # Convert input data to numpy arrays
    x = np.array(x, dtype=np.float32)
    a_m = np.array(a_m, dtype=np.float32)
    t_values = np.array(t_values, dtype=np.int64)
    d_values = np.array(d_values, dtype=np.int32)

    # Get the length of a_m
    half_len = len(a_m) // 2

    # Initialize the Tg matrix with zeros
    Tg = np.zeros((len(t_values), len(d_values)), dtype=np.float32)

    # Iterate over the d_values and calculate the Tg matrix
    for j, d in enumerate(d_values):
        # Initialize Tg column for this d value
        Tg_col = np.zeros(len(t_values), dtype=np.float32)

        for m in range(-half_len, half_len + 1):
            # Compute the indices for the current (t, d) pair
            indices = t_values + m * d

            # Select the valid indices (within bounds of x)
            valid_mask = (indices >= 0) & (indices < len(x))
            valid_t_values = t_values[valid_mask]
            valid_indices = indices[valid_mask]

            # Accumulate the contribution to the Tg matrix
            Tg_col[valid_t_values] += a_m[half_len + m] * x[valid_indices]

        # Store the result for this d value
        Tg[:, j] = Tg_col

    return Tg



def calculate_Tg_torch(x, t_values, d_values, a_m, device='cuda'):
    """
    Calculates the Tg matrix for given input data and parameters using PyTorch and CUDA.
    """

    torch.cuda.empty_cache()
    
    # Set x to device
    x = torch.tensor(x, device=device, dtype=torch.float32)
    a_m = torch.tensor(a_m, device=device, dtype=torch.float32)
    t_values = torch.tensor(t_values, device=device, dtype=torch.int64)
    d_values = torch.tensor(d_values, device=device, dtype=torch.int32)
    
    # Initialize the Tg matrix with zeros
    Tg = torch.zeros((len(t_values), len(d_values)), device=device, dtype=torch.float32)

    # Get the length of a_m
    half_len = len(a_m) // 2
    
    # Iterate over the d_values and calculate the Tg matrix
    for j, d in enumerate(d_values):
        # Initialize Tg column for this d value
        Tg_col = torch.zeros(len(t_values), device=device, dtype=torch.float32)

        for m in range(-half_len, half_len + 1):
            # Calculate the indices to use for x
            indices = t_values + m * d

            # Select the valid indices (within bounds of x)
            valid_indices = (indices >= 0) & (indices < len(x))
            valid_t_values = t_values[valid_indices]
            valid_indices = indices[valid_indices]

            # Accumulate the contribution to the Tg matrix
            Tg_col[valid_t_values] += a_m[half_len + m] * x[valid_indices]
        
        # Store the result for this d value
        Tg[:, j] = Tg_col

    return Tg.cpu().numpy()

def cegmil_tempogram(dir=None, onset_envelope=None, fps=100, d_range=(1, 300), 
                     a_m=[0.25, 0.5, 1, 0.5, 0.25], 
                     device='cpu', plot=False, ground=None,):
    """
    Calculate the Cemgil Tempogram.

    > Cemgil A T, Kappen B, Desain P, et al. On tempo tracking: Tempogram representation and Kalman filtering[J]. Journal of New Music Research, 2000, 29(4): 259-273.
    
    Args:
        dir (str, optional): The directory of the audio file. Either `dir` or `onset_envelope` must be provided. Defaults to None.
        onset_envelope (array-like, optional): The onset envelope of the audio signal. Either `dir` or `onset_envelope` must be provided. Defaults to None.
        fps (int, optional): Frames per second. Defaults to 100.
        d_range (tuple, optional): The range of d values. Defaults to (1, 300).
        a_m (array-like, optional): The array of a_m values. Defaults to [0.25, 0.5, 1, 0.5, 0.25].
        device (str, optional): The device to use for calculation. Either 'cpu' or 'cuda'. Defaults to 'cpu'.
        plot (bool, optional): Whether to plot the tempogram. Defaults to False.
        ground (array-like, optional): The ground truth beats. Defaults to None.

    Returns:
        array-like: The Cemgil Tempogram. Shape = (d, times)
    """
    
    if onset_envelope is None:
        if dir is None:
            raise "Either dir or onset_envelope must be provided"
        # get onset strength signal using madmom
        from madmom.features.beats import RNNBeatProcessor 
        onset_envelope = RNNBeatProcessor(fps=fps)(dir)

    x = onset_envelope
    x = np.array(x) / np.sum(x)

    t_values = np.arange(len(x))
    d_values = np.arange(d_range[0], d_range[1] + 1)
    a_m =  np.array(a_m) / np.sum(a_m)
    
    if device == 'cuda' and torch.cuda.is_available():
        Tg = calculate_Tg_torch(x, t_values, d_values, a_m)
    elif device == 'cpu':
        Tg = calculate_Tg_numpy(x, t_values, d_values, a_m)
    else:
        raise ValueError("Invalid device or CUDA is not available")

    if plot:
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 3), dpi=200)
        gs = fig.add_gridspec(2, 2, width_ratios=(1, 6), height_ratios=(4, 1), hspace=0.0, wspace=0.0)
        ax_main = fig.add_subplot(gs[0, 1])
        ax_d_marginal = fig.add_subplot(gs[0, 0], sharey=ax_main)
        ax_t_marginal = fig.add_subplot(gs[1, 1], sharex=ax_main)
        
        im = ax_main.imshow(Tg.T, extent=[0, len(t_values)/fps, d_range[0]/fps, d_range[1]/fps], 
                            aspect='auto', origin='lower', cmap='Blues')
        
        ax_main.set_title('p(d,t)')
        ax_main.grid(False)

        if ground is not None:
            beats = ground[ground < len(x)/fps]
            intervals = np.diff(beats)
            ax_main.scatter(beats[:-1], intervals, c='g', s=10, alpha=0.5, label='ground truth')

        t_marginal = np.sum(np.exp(Tg*100+1), axis=1)
        ax_t_marginal.plot(np.arange(len(t_values))/fps, t_marginal)
        ax_t_marginal.set_xlabel('p(t)')
        ax_t_marginal.grid(linestyle='--')
        ax_t_marginal.set_yticks([])       
        
        pd_x = np.sum(np.exp(Tg*100+1), axis=0)
        ax_d_marginal.plot(pd_x, np.arange(d_range[0], d_range[1] + 1)/fps)
        ax_d_marginal.set_ylabel('p(d|t)')
        ax_d_marginal.grid(linestyle='--')
        ax_d_marginal.set_xticks([])
        plt.tight_layout()

        plt.show()
    
    # align with librosa, shape = (d, t)
    return Tg.T



if __name__ == "__main__":

    # simple usage:
    Tg = cegmil_tempogram('test.wav', plot=True)

    # Compare numpy and torch implementations
    from time import time
    start = time()
    Tg = cegmil_tempogram('test.wav', device='cpu')
    print(f'Numpy version: {time()-start}s')

    start = time()
    Tg_torch = cegmil_tempogram('test.wav', device='cuda')
    print(f'Torch version: {time()-start}s')

    # Check if the results are the same
    if np.allclose(Tg, Tg_torch, atol=1e-6):
        print("Results are the same.")


