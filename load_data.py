import glob, random
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
TRN_CELL_IDS = ['G1', 'V5', 'W4', 'W5', 'W8', 'W9']
TEST_CELL_IDS = ['V4', 'W10']


class Stanford_Degradation_Dataset(Dataset):
    def __init__(self, data_dir, mode='train', detrend=True, seq_len=10, aug_ratio=64):
        """
        detrend(bool): True for detrended data
        seq_len(int): select the number of cycles from the full dataset
        aug_ratio(int): sample aug_ratio times for full cycles
        """
        if mode != 'train' and mode != 'valid':
            raise ValueError('mode must be train or valid')
        
        self.data = []
        if detrend:
            self.mean, self.std = [0, -0.55], [0.033, 1.1] # mean, std for detrended voltage, current
        else:
            self.mean, self.std = [3.6, -4.8], [0.13, 6.1] # mean, std for raw voltage, current

        cycling_data = load_Stanford_data(data_dir, mode, detrend)
        
        for seq in cycling_data:
            seq = seq.copy()
            if len(seq)<seq_len: continue # skip when cycle number is less than seq_len

            for i in range(2):
                seq[:, i, :] = (seq[:, i, :]-self.mean[i])/self.std[i]
            for i in range(aug_ratio):
                rnd_id = np.random.choice(len(seq), seq_len, replace=False) # select random cycles
                self.data.append(seq[np.sort(rnd_id)])
        self.data = np.array(self.data)
        print(self.data.shape)

    def __getitem__(self, index):
        x = self.data[index]
        return x  # shape = (batch_size, n_cycles, n_features, time_step)

    def __len__(self):
        return len(self.data)
    
    def normalize_coef(self):
        return self.mean, self.std


class Stanford_Capacity_Dataset(Dataset):
    def __init__(self, signal, capacity, n_features=2):
        signal = np.concatenate(signal, axis=0)
        self.target = (np.concatenate(capacity, axis=0)-4.64)/0.12
        # print(signal.shape, self.target.shape)
        self.mean, self.std = [0, -0.55], [0.033, 1.1] # for filtered data

        for j in range(n_features):
            signal[:, j, :] = (signal[:, j, :]-self.mean[j])/self.std[j]
        self.features = np.expand_dims(signal, axis=1)
        self.target = np.expand_dims(self.target, axis=1)
        
    def __getitem__(self, index):
        features = self.features[index]
        target = self.target[index]
        return features, target

    def __len__(self):
        return len(self.features)


def load_Stanford_data(data_dir, mode='train', detrend=True, seed=None, full=False):
    """
    load the voltage and current information in cycling test

    Args:
        data_dir (str): The directory containing processed cycling npy files.
        mode (str): Mode of data to load ('train' or 'valid').
        detrend (bool): Load detrended or raw data.
        seed (int): Random seed for reproducibility.
        full (bool): Load all the segment or just a subset. only available for detrend=True

    Returns:
        List[np.ndarray]: A list of numpy arrays containing the VI data for each cell. 
                          if full: (n_cycle, n_segment: 20, n_feature: 2, time_step: 3600).
                          else: (n_cycle, n_feature: 2, time_step: 3600).
    """
    if mode != 'train' and mode != 'valid':
        raise ValueError('mode must be train or valid')
    
    if mode=='train':
        cell_ids = TRN_CELL_IDS
    elif mode=='valid':
        cell_ids = TEST_CELL_IDS

    if seed is not None:
        random.seed(seed)

    vi_per_cell = []
    if detrend:
        for cell in tqdm(cell_ids, desc=f'load cell ({mode})'):
            filenames = sorted(glob.glob(f'{data_dir}/{cell}/*.npy'))
            vi_per_cycle = []
            for f in filenames:
                if full: # the shape of vi_per_cycle: (n_cycle, 2, 3600)
                    vi_per_cycle.append(np.load(f))
                else: # the shape of vi_per_cycle: (n_cycle, 20, 2, 3600)
                    segment_id = random.randint(0, 19)
                    vi_per_cycle.append(np.load(f)[segment_id])
            vi_per_cell.append(np.array(vi_per_cycle))
    else:
        for cell in tqdm(cell_ids, desc=f'load cell ({mode})'):
            filenames = sorted(glob.glob(f'{data_dir}/{cell}/*.npy'))
            vi_per_cycle = []
            for f in filenames:
                vi = np.load(f).transpose((1, 0)) # (n_features, time_step)
                start = random.randint(0, vi.shape[1]-3601) # randomly select a patch with 1-hour length
                vi_per_cycle.append(vi[:, start:start+3600]) 
            vi_per_cell.append(np.array(vi_per_cycle))

    return vi_per_cell


def load_Stanford_capacity_data(data_dir='Stanford_Dataset/filtered_info_vi', mode='train'):
    def find_nearest(array, value):
        array = np.asarray(array)
        if np.abs(array - value).min()<=10:
            return np.abs(array - value).argmin()
        return False
    
    if mode != 'train' and mode != 'valid':
        raise ValueError('mode must be train or valid')
    
    if mode=='train':
        cell_ids = TRN_CELL_IDS
    elif mode=='valid':
        cell_ids = TEST_CELL_IDS

    vi_per_cell, cap_per_cell, cycle_per_cycle = [], [], []
    for cell in cell_ids:
        filenames = sorted(glob.glob(f'{data_dir}/{cell}/*.npy'))
        cycle_id = np.sort(np.array([int(f[-7:-4].lstrip('0')) for f in filenames])) # extract correspond cycle id
        capacity_test = np.load(f'Stanford_Dataset/capacity_each_cell/{cell}_capacity.npy') # load real capacity
        test_cycle_id = capacity_test[0, :] # cycle id with measured capacity

        vi_per_cycle, cap_per_cycle, cycle_list = [], [], []
        for i, cycle in enumerate(test_cycle_id):
            cycle = 1 if cycle == 0 else cycle
            target_cycle = find_nearest(cycle_id, cycle)
            if target_cycle: 
                v_t = np.load(filenames[target_cycle])
            else: 
                continue
            vi_per_cycle.append(v_t)
            cap_per_cycle.append(capacity_test[1, i])
            cycle_list.append(capacity_test[0, i])

        vi_per_cell.append(np.array(vi_per_cycle))
        cap_per_cell.append(np.array(cap_per_cycle))
        cycle_per_cycle.append(np.array(cycle_list))

    return vi_per_cell, cap_per_cell, cycle_per_cycle


if __name__=='__main__':
    data = Stanford_Degradation_Dataset('Stanford_Dataset/detrended_VI')