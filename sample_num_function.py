import numpy as np

def sample_number_function(N_sample, sample_times, first_sample, n_sample, sample_mode):
    '''
    N_sample:total sample number
    first_sample: seita_0
    n_sample: seita_n

    return sampla_strategy: np.array
    '''

    import math

    first_sample = first_sample*N_sample
    n_sample_num = n_sample*N_sample
    
    sample_strategy = np.zeros(sample_times)
    # print(sample_strategy)
    sample_strategy[0] = first_sample
    # print(sample_strategy)

    if sample_mode == "linear":
        d = int(2*(n_sample_num - sample_times*first_sample)/(sample_times*(sample_times-1)))
        # print(d)
        for i in range(1, sample_times):
            sample_strategy[i] = sample_strategy[i-1] + d
        return sample_strategy
    elif sample_mode == "convex":
        for sample_times in reversed(range(1, sample_times)):
            if 6*(sample_times*first_sample - n_sample_num)/(sample_times*(sample_times-1)*(2*sample_times-1)) > (sample_times-1)**2/first_sample :
                break
        d = (9*(sample_times*first_sample - n_sample_num)**2)/(4*(sample_times-1)**3)
        # print(d)
        for i in range(1, sample_times):
            sample_strategy[i] = int(first_sample - (d*i)**0.5)
        return sample_strategy
    elif sample_mode == "square":
        for sample_times in reversed(range(1, sample_times)):
            if 6*(sample_times*first_sample - n_sample_num)/(sample_times*(sample_times-1)*(2*sample_times-1)) > (sample_times-1)**2/first_sample :
                break
        d = 6*(sample_times*first_sample - n_sample_num)/(sample_times*(sample_times-1)*(2*sample_times-1))
        # print(d)
        for i in range(1, sample_times):
            sample_strategy[i] = int(first_sample - d*i**2)
        return sample_strategy
        
if __name__ == '__main__':

    N_sample = 100
    sample_times = 3
    first_sample = 0.5
    n_sample = 0.7
    sample_mode = "convex"
    
    print(sample_number_function(N_sample, sample_times, first_sample, n_sample, sample_mode))

