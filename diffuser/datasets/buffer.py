import numpy as np
import random
import torch
def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

from collections import deque


class ReplayBufferSAS:    # state-action-state buffer, trained for cost guide model
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)  # 自动限制长度
        self.buffer_size = buffer_size

    def add(self, data, label):
        self.buffer.append((data.float(), label.float()))  # 直接追加

    '''def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([transition[0] for transition in batch])  # 批量转换为 numpy
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        return states, actions, rewards, next_states, dones'''

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate data and label lists
        data_samples, label_samples = zip(*batch)
        
        # Stack into tensors with proper shapes
        data_batch = torch.stack(data_samples)  # shape: (batch_size, 12)
        label_batch = torch.stack(label_samples)  # shape: (batch_size, 1)
        return data_batch, label_batch

    def __len__(self):  # <-- Add this method to support len()
        return len(self.buffer)

    def save(self, path):
        """保存buffer到文件"""
        # 转换为list以便保存（deque不能直接保存）
        buffer_list = list(self.buffer)
        torch.save({
            'buffer': buffer_list,
            'buffer_size': self.buffer_size
        }, path)

    def load(self, path):
        """将保存的数据直接加载到当前buffer中（不清空现有数据）"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No buffer file at {path}")
        
        # 加载保存的数据
        checkpoint = torch.load(path)
        
        # 检查buffer大小是否兼容
        if checkpoint['buffer_size'] != self.buffer.maxlen:
            print(f"Warning: Saved buffer size {checkpoint['buffer_size']} "
                f"differs from current size {self.buffer.maxlen}")
        
        # 将数据追加到当前buffer
        for data, label in checkpoint['buffer']:
            self.buffer.append((data, label))
        
        #print(f"Loaded {len(checkpoint['buffer'])} samples into existing buffer")
