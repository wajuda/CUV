import torch
import torch.nn as nn
import pdb
import numpy as np
import diffuser.utils as utils
 
# 目前的想法是输入地图和当前位置算出cost,返回grad.
 
from scipy.ndimage import distance_transform_edt
from diffuser.models import MultiLinearLayer

class ValueGuide_maze2d(nn.Module):
    def __init__(self, maze_layout=None, normalizer=None, device="cuda", action_dim=2, state_dim=2):
        super().__init__()
        # 迷宫布局定义
        self.maze_layout = maze_layout or \
            "############\\"+\
            "#OOOO#OOOOO#\\"+\
            "#O##O#O#O#O#\\"+\
            "#OOOOOO#OOO#\\"+\
            "#O####O###O#\\"+\
            "#OO#O#OOOOO#\\"+\
            "##O#O#O#O###\\"+\
            "#OO#OOO#OGO#\\"+\
            "############"
        
        # 解析迷宫并预计算距离场
        self.device = device
        self.directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=self.device)

        
        self.action_dim = action_dim
        self.state_dim = state_dim
        #self.distance_map = self._precompute_distance_map()
        self.normalizer = normalizer
        # 注册为buffer以支持GPU
        #self.register_buffer("distance_map_tensor", 
        #                  torch.from_numpy(self.distance_map).float())
        self.maze_grid = self._parse_maze()
        self.height, self.width = self.maze_grid.shape
        

        self.bound_min = torch.tensor([0, 0], device=self.device)
        self.bound_max = torch.tensor([self.height-1, self.width-1], device =self.device)
        self._precompute_neighbor_masks()
        self.to(device)
    def _parse_maze(self):
        rows = self.maze_layout.split('\\')
        maze_np = np.array([[1 if c == '#' else 0 for c in row] for row in rows])
        return torch.from_numpy(maze_np).to(self.device)  
 

    def _precompute_neighbor_masks(self):
        neighbor_masks = torch.zeros((self.height, self.width, 4), dtype=torch.bool, device=self.device)
        #directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=self.device)

        for i in range(self.height):
            for j in range(self.width):
                neighbors = (torch.tensor([i, j], device = self.device)+self.directions).clip(self.bound_min, self.bound_max)

                mask = self.maze_grid[neighbors[:, 0], neighbors[:, 1]] == 1
                neighbor_masks[i, j] = mask

        #self.neighbor_masks = torch.stack(neighbour_masks).to(self.device)
        self.neighbor_masks = neighbor_masks
 
    def _compute_values_batch(self, grid_coords, indices):
        batch_size, horizon, _ = grid_coords.shape  # B * H * 2

        directions = self.directions.unsqueeze(0).unsqueeze(0)  # 1 * 1 * 4 * 2
        neighbor_masks = self.neighbor_masks[indices[:, :, 0], indices[:, :, 1]] # B * H * 4

        neighbors = indices.unsqueeze(2) + self.directions  # B * H * 4 * 2
        neighbors = neighbors.clip(self.bound_min, self.bound_max).float()
        '''print(neighbors)
        print(neighbor_masks)'''

        distances = torch.norm(neighbors - grid_coords.unsqueeze(-2), dim=-1)  # B * H * 4
        masked_distances = (distances * neighbor_masks.float()).sum(dim=-1) / (neighbor_masks.sum(dim=-1) + 1e-6)  # B * H * 4

        #values = -masked_distances.sum(dim=-1)  # B * H
        return masked_distances.mean(dim=-1)  # B

    def forward(self, trajectories, cond=None, t=None):
        """
        向量化计算轨迹价值
        Input:  trajectories - [batch_size, horizon, 2] (坐标范围[0,1])
        Output: valu es - [batch_size]
        """
        # 将坐标映射到网格索引 [-1,1] -> [0, width-1]
        '''sequences = self.normalizer.unnormalize(utils.to_np(trajectories[:,:,self.action_dim:]), 'observations')
        grid_coords = torch.from_numpy(sequences[:,:,:self.state_dim]).to(self.device)'''
        #grid_coords = trajectories[:,:,self.action_dim:].to(self.device)
        sequences = self.normalizer.unnormalize(trajectories[:,:,self.action_dim:], 'observations')
        grid_coords = sequences[:,:,:self.state_dim].to(self.device)
        
        
        # 四舍五入并转为整数索引 (需要clamp防止越界)
        indices = torch.round(grid_coords).long().to(self.device)
        #print(indices)
        indices = indices.clip(self.bound_min, self.bound_max)
        #print(indices)

        values = self._compute_values_batch(grid_coords, indices)
        
        

        # 对每条轨迹取时间维度的平均值 [batch_size]
        return values

    def gradients(self, x, *args):
        """计算value对输入的梯度（自动向量化）"""
        x = x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad(y.sum(), x, create_graph=False)[0]
        return y, grad

class ValueGuide_maze2d_v2(nn.Module): #八邻居，-1/x
    def __init__(self, maze_layout=None, normalizer=None, device="cuda", action_dim=2, state_dim=2):
        super().__init__()
        # 迷宫布局定义
        self.maze_layout = maze_layout or \
            "############\\"+\
            "#OOOO#OOOOO#\\"+\
            "#O##O#O#O#O#\\"+\
            "#OOOOOO#OOO#\\"+\
            "#O####O###O#\\"+\
            "#OO#O#OOOOO#\\"+\
            "##O#O#O#O###\\"+\
            "#OO#OOO#OGO#\\"+\
            "############"
        
        # 解析迷宫并预计算距离场
        self.device = device
        self.directions = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]], device=self.device)
        self.bias = 1
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        #self.distance_map = self._precompute_distance_map()
        self.normalizer = normalizer
        # 注册为buffer以支持GPU
        #self.register_buffer("distance_map_tensor", 
        #                  torch.from_numpy(self.distance_map).float())
        self.maze_grid = self._parse_maze()
        self.height, self.width = self.maze_grid.shape
        

        self.bound_min = torch.tensor([0, 0], device=self.device)
        self.bound_max = torch.tensor([self.height-1, self.width-1], device =self.device)
        self._precompute_neighbor_masks()
        self.to(device)
    def _parse_maze(self):
        rows = self.maze_layout.split('\\')
        maze_np = np.array([[1 if c == '#' else 0 for c in row] for row in rows])
        return torch.from_numpy(maze_np).to(self.device)  
 

    def _precompute_neighbor_masks(self):
        neighbor_masks = torch.zeros((self.height, self.width, 8), dtype=torch.bool, device=self.device)

        for i in range(self.height):
            for j in range(self.width):
                neighbors = (torch.tensor([i, j], device = self.device)+self.directions).clip(self.bound_min, self.bound_max)
                #print(neighbors.shape)
                mask = self.maze_grid[neighbors[:, 0], neighbors[:, 1]] == 1
                #print(mask.shape)
                neighbor_masks[i, j] = mask
                #print(neighbor_masks.shape)

        #self.neighbor_masks = torch.stack(neighbour_masks).to(self.device)
        self.neighbor_masks = neighbor_masks
 
    def _compute_values_batch(self, grid_coords, indices):
        batch_size, horizon, _ = grid_coords.shape  # B * H * 2

        directions = self.directions.unsqueeze(0).unsqueeze(0)  # 1 * 1 * 8 * 2
        neighbor_masks = self.neighbor_masks[indices[:, :, 0], indices[:, :, 1]] # B * H * 8

        neighbors = indices.unsqueeze(2) + self.directions  # B * H * 8 * 2
        neighbors = neighbors.clip(self.bound_min, self.bound_max).float()
        '''print(neighbors)
        print(neighbor_masks)'''

        distances = torch.norm(neighbors - grid_coords.unsqueeze(-2), dim=-1)  # B * H * 4
        values = -1 / (distances + self.bias)
        #masked_distances = (distances * neighbor_masks.float()).sum(dim=-1) / (neighbor_masks.sum(dim=-1) + 1e-6)  # B * H * 4
        masked_values = (values * neighbor_masks.float()).sum(dim=-1) / (neighbor_masks.sum(dim=-1) + 1e-6)  # B * H * 4

        #values = -masked_distances.sum(dim=-1)  # B * H
        return masked_values.mean(dim=-1)  # B

    def forward(self, trajectories, cond=None, t=None):
        """
        向量化计算轨迹价值
        Input:  trajectories - [batch_size, horizon, 2] (坐标范围[0,1])
        Output: valu es - [batch_size]
        """
        # 将坐标映射到网格索引 [-1,1] -> [0, width-1]
        '''sequences = self.normalizer.unnormalize(utils.to_np(trajectories[:,:,self.action_dim:]), 'observations')
        grid_coords = torch.from_numpy(sequences[:,:,:self.state_dim]).to(self.device)'''
        #grid_coords = trajectories[:,:,self.action_dim:].to(self.device)
        sequences = self.normalizer.unnormalize(trajectories[:,:,self.action_dim:], 'observations')
        grid_coords = sequences[:,:,:self.state_dim].to(self.device)
        
        
        # 四舍五入并转为整数索引 (需要clamp防止越界)
        indices = torch.round(grid_coords).long().to(self.device)
        #print(indices)
        indices = indices.clip(self.bound_min, self.bound_max)
        #print(indices)

        values = self._compute_values_batch(grid_coords, indices)
        
        

        # 对每条轨迹取时间维度的平均值 [batch_size]
        return values

    def gradients(self, x, *args):
        """计算value对输入的梯度（自动向量化）"""
        x = x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad(y.sum(), x, create_graph=False)[0]
        return y, grad

class ValueGuide_maze2d_v3(nn.Module): #八邻居，-1/x
    def __init__(self, maze_layout=None, normalizer=None, device="cuda", action_dim=2, state_dim=2):
        super().__init__()
        # 迷宫布局定义
        self.maze_layout = maze_layout or \
            "############\\"+\
            "#OOOO#OOOOO#\\"+\
            "#O##O#O#O#O#\\"+\
            "#OOOOOO#OOO#\\"+\
            "#O####O###O#\\"+\
            "#OO#O#OOOOO#\\"+\
            "##O#O#O#O###\\"+\
            "#OO#OOO#OGO#\\"+\
            "############"
        
        # 解析迷宫并预计算距离场
        self.device = device
        #self.directions = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]], device=self.device)
        #self.directions = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]], device=self.device)
        self.directions = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [0, 0]], device=self.device)
        self.direction_weights = torch.tensor([1,1,1,1,1,1,1,1,0.25], device = self.device)
        
        self.bias = 1
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        #self.distance_map = self._precompute_distance_map()
        self.normalizer = normalizer
        # 注册为buffer以支持GPU
        #self.register_buffer("distance_map_tensor", 
        #                  torch.from_numpy(self.distance_map).float())
        self.maze_grid = self._parse_maze()
        self.height, self.width = self.maze_grid.shape
        

        self.bound_min = torch.tensor([0, 0], device=self.device)
        self.bound_max = torch.tensor([self.height-1, self.width-1], device =self.device)
        self._precompute_neighbor_masks()
        self.to(device)
    def _parse_maze(self):
        rows = self.maze_layout.split('\\')
        maze_np = np.array([[1 if c == '#' else 0 for c in row] for row in rows])
        return torch.from_numpy(maze_np).to(self.device)  
 

    def _precompute_neighbor_masks(self):
        neighbor_masks = torch.zeros((self.height, self.width, self.directions.shape[0]), dtype=torch.bool, device=self.device)

        for i in range(self.height):
            for j in range(self.width):
                neighbors = (torch.tensor([i, j], device = self.device)+self.directions).clip(self.bound_min, self.bound_max)
                #print(neighbors.shape)
                mask = self.maze_grid[neighbors[:, 0], neighbors[:, 1]] == 1
                #print(mask.shape)
                neighbor_masks[i, j] = mask
                #print(neighbor_masks.shape)

        #self.neighbor_masks = torch.stack(neighbour_masks).to(self.device)
        self.neighbor_masks = neighbor_masks
 
    def _compute_values_batch(self, grid_coords, indices):
        batch_size, horizon, _ = grid_coords.shape  # B * H * 2

        directions = self.directions.unsqueeze(0).unsqueeze(0)  # 1 * 1 * 8 * 2
        neighbor_masks = self.neighbor_masks[indices[:, :, 0], indices[:, :, 1]] # B * H * 8

        neighbors = indices.unsqueeze(2) + self.directions  # B * H * 8 * 2
        neighbors = neighbors.clip(self.bound_min, self.bound_max).float()
        '''print(neighbors)
        print(neighbor_masks)'''

        distances = torch.norm(neighbors - grid_coords.unsqueeze(-2), dim=-1)  # B * H * 4
        values = -1 / (distances + self.bias)
        #masked_distances = (distances * neighbor_masks.float()).sum(dim=-1) / (neighbor_masks.sum(dim=-1) + 1e-6)  # B * H * 4
        masked_values = (values * neighbor_masks.float()*self.direction_weights.float()).sum(dim=-1) / (neighbor_masks.sum(dim=-1) + 1e-6)  # B * H * 4

        #values = -masked_distances.sum(dim=-1)  # B * H
        return masked_values.mean(dim=-1)  # B

    def forward(self, trajectories, cond=None, t=None):
        """
        向量化计算轨迹价值
        Input:  trajectories - [batch_size, horizon, 2] (坐标范围[0,1])
        Output: valu es - [batch_size]
        """
        # 将坐标映射到网格索引 [-1,1] -> [0, width-1]
        '''sequences = self.normalizer.unnormalize(utils.to_np(trajectories[:,:,self.action_dim:]), 'observations')
        grid_coords = torch.from_numpy(sequences[:,:,:self.state_dim]).to(self.device)'''
        #grid_coords = trajectories[:,:,self.action_dim:].to(self.device)
        sequences = self.normalizer.unnormalize(trajectories[:,:,self.action_dim:], 'observations')
        grid_coords = sequences[:,:,:self.state_dim].to(self.device)
        
        
        # 四舍五入并转为整数索引 (需要clamp防止越界)
        indices = torch.round(grid_coords).long().to(self.device)
        #print(indices)
        indices = indices.clip(self.bound_min, self.bound_max)
        #print(indices)

        values = self._compute_values_batch(grid_coords, indices)
        
        

        # 对每条轨迹取时间维度的平均值 [batch_size]
        return values

    def gradients(self, x, *args):
        """计算value对输入的梯度（自动向量化）"""
        x = x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad(y.sum(), x, create_graph=False)[0]
        return y, grad
    

class ValueGuide_maze2d_v4(nn.Module): #r(x,x')对x'求导，x'相对于x离目标越近越好
    def __init__(self, maze_layout=None, normalizer=None, device="cuda", action_dim=2, state_dim=2):
        super().__init__()
        # 迷宫布局定义
        self.maze_layout = maze_layout or \
            "############\\"+\
            "#OOOO#OOOOO#\\"+\
            "#O##O#O#O#O#\\"+\
            "#OOOOOO#OOO#\\"+\
            "#O####O###O#\\"+\
            "#OO#O#OOOOO#\\"+\
            "##O#O#O#O###\\"+\
            "#OO#OOO#OGO#\\"+\
            "############"
        
        # 解析迷宫并预计算距离场
        self.device = device
        self.directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=self.device)
        self.goal = torch.tensor([7, 9], device=self.device)  # 假设目标位置为(7, 7)

        
        self.action_dim = action_dim
        self.state_dim = state_dim
        #self.distance_map = self._precompute_distance_map()
        self.normalizer = normalizer
        # 注册为buffer以支持GPU
        #self.register_buffer("distance_map_tensor", 
        #                  torch.from_numpy(self.distance_map).float())
        self.maze_grid = self._parse_maze()
        self.height, self.width = self.maze_grid.shape
        

        self.bound_min = torch.tensor([0, 0], device=self.device)
        self.bound_max = torch.tensor([self.height-1, self.width-1], device =self.device)
        self._precompute_neighbor_masks()
        self.to(device)
    def _parse_maze(self):
        rows = self.maze_layout.split('\\')
        maze_np = np.array([[1 if c == '#' else 0 for c in row] for row in rows])
        return torch.from_numpy(maze_np).to(self.device)  
 

    def _precompute_neighbor_masks(self):
        neighbor_masks = torch.zeros((self.height, self.width, 4), dtype=torch.bool, device=self.device)
        #directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=self.device)

        for i in range(self.height):
            for j in range(self.width):
                neighbors = (torch.tensor([i, j], device = self.device)+self.directions).clip(self.bound_min, self.bound_max)

                mask = self.maze_grid[neighbors[:, 0], neighbors[:, 1]] == 1
                neighbor_masks[i, j] = mask

        #self.neighbor_masks = torch.stack(neighbour_masks).to(self.device)
        self.neighbor_masks = neighbor_masks
 
    def _compute_values_batch(self, grid_coords, indices):
        batch_size, horizon, _ = grid_coords.shape  # B * H * 2

        directions = self.directions.unsqueeze(0).unsqueeze(0)  # 1 * 1 * 4 * 2
        neighbor_masks = self.neighbor_masks[indices[:, :, 0], indices[:, :, 1]] # B * H * 4

        neighbors = indices.unsqueeze(2) + self.directions  # B * H * 4 * 2
        neighbors = neighbors.clip(self.bound_min, self.bound_max).float()
        '''print(neighbors)
        print(neighbor_masks)'''

        distances = torch.norm(neighbors - grid_coords.unsqueeze(-2), dim=-1)  # B * H * 4
        masked_distances = (distances * neighbor_masks.float()).sum(dim=-1) / (neighbor_masks.sum(dim=-1) + 1e-6)  # B * H * 4

        #values = -masked_distances.sum(dim=-1)  # B * H
        return masked_distances.mean(dim=-1)  # B

    def forward(self, trajectories, cond=None, t=None):
        """
        向量化计算轨迹价值
        Input:  trajectories - [batch_size, horizon, 2] (坐标范围[0,1])
        Output: valu es - [batch_size]
        """
        # 将坐标映射到网格索引 [-1,1] -> [0, width-1]
        '''sequences = self.normalizer.unnormalize(utils.to_np(trajectories[:,:,self.action_dim:]), 'observations')
        grid_coords = torch.from_numpy(sequences[:,:,:self.state_dim]).to(self.device)'''
        #grid_coords = trajectories[:,:,self.action_dim:].to(self.device)
        sequences = self.normalizer.unnormalize(trajectories[:,:,self.action_dim:], 'observations')
        grid_coords = sequences[:,:,:self.state_dim].to(self.device)
        #grid_coords = trajectories.to(self.device)
        self.goal = grid_coords[:, -1, :].to(self.device)  # 使用轨迹的最后一个位置作为目标
        
        current_coords = grid_coords[:, :-1, :].detach()  # 去掉最后一个位置
        next_coords = grid_coords[:, 1:, :]  # 去掉第一个位置

        current_diff = torch.norm(current_coords - self.goal.unsqueeze(1), dim=-1)  # 计算每个位置到目标的距离
        next_diff = torch.norm(next_coords - self.goal.unsqueeze(1), dim=-1)  # 计算每个位置到目标的距离

        value_diff = current_diff - next_diff  # 计算每个位置的价值差
        value_diff = torch.clamp(value_diff, max=0)
        # 只对 next_coords 求均值（batch 维度），忽略 current_coords
        values = value_diff.mean(dim=1)

        # 对每条轨迹取时间维度的平均值 [batch_size]
        return values

    def gradients(self, x, *args):
        """计算value对输入的梯度（自动向量化）"""
        x = x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad(y.sum(), x, create_graph=False)[0]
        return y, grad

class ValueGuide_maze2d_v5(nn.Module): #r(x,x')对x'求导，x'相对于x离目标越近越好
    def __init__(self, maze_layout=None, normalizer=None, device="cuda", action_dim=2, state_dim=2):
        super().__init__()
        # 迷宫布局定义
        self.maze_layout = maze_layout or \
            "############\\"+\
            "#OOOO#OOOOO#\\"+\
            "#O##O#O#O#O#\\"+\
            "#OOOOOO#OOO#\\"+\
            "#O####O###O#\\"+\
            "#OO#O#OOOOO#\\"+\
            "##O#O#O#O###\\"+\
            "#OO#OOO#OGO#\\"+\
            "############"
        
        # 解析迷宫并预计算距离场
        self.device = device
        self.directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=self.device)
        self.goal = torch.tensor([7, 9], device=self.device)  # 假设目标位置为(7, 7)

        
        self.action_dim = action_dim
        self.state_dim = state_dim
        #self.distance_map = self._precompute_distance_map()
        self.normalizer = normalizer
        # 注册为buffer以支持GPU
        #self.register_buffer("distance_map_tensor", 
        #                  torch.from_numpy(self.distance_map).float())
        self.maze_grid = self._parse_maze()
        self.height, self.width = self.maze_grid.shape
        

        self.bound_min = torch.tensor([0, 0], device=self.device)
        self.bound_max = torch.tensor([self.height-1, self.width-1], device =self.device)
        self._precompute_neighbor_masks()
        self.to(device)
    def _parse_maze(self):
        rows = self.maze_layout.split('\\')
        maze_np = np.array([[1 if c == '#' else 0 for c in row] for row in rows])
        return torch.from_numpy(maze_np).to(self.device)  
 

    def _precompute_neighbor_masks(self):
        neighbor_masks = torch.zeros((self.height, self.width, 4), dtype=torch.bool, device=self.device)
        #directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=self.device)

        for i in range(self.height):
            for j in range(self.width):
                neighbors = (torch.tensor([i, j], device = self.device)+self.directions).clip(self.bound_min, self.bound_max)

                mask = self.maze_grid[neighbors[:, 0], neighbors[:, 1]] == 1
                neighbor_masks[i, j] = mask

        #self.neighbor_masks = torch.stack(neighbour_masks).to(self.device)
        self.neighbor_masks = neighbor_masks
 
    def _compute_values_batch(self, grid_coords, indices):

       

       
        ###


        batch_size, horizon, _ = grid_coords.shape  # B * H * 2

        current_coords = grid_coords[:, :-1, :].detach()  # 去掉最后一个位置
        next_coords = grid_coords[:, 1:, :]  # 去掉第一个位置
        next_indices = indices[:, 1:, :].detach()  # 去掉第一个位置的索引

        masks = self.maze_grid[next_indices[:, :, 0], next_indices[:, :, 1]]

        current_diff = torch.norm(current_coords - next_indices, dim=-1)  # 计算每个位置到目标的距离
        next_diff = torch.norm(next_coords - next_indices, dim=-1)  # 计算每个位置到目标的距离

        value_diff =  next_diff - current_diff # 计算每个位置的价值差
        value_diff = torch.clamp(value_diff, max=0)
        # 只对 next_coords 求均值（batch 维度），忽略 current_coords
        masked_value_diff = value_diff * masks.float()

        #values = -masked_distances.sum(dim=-1)  # B * H
        return masked_value_diff.mean(dim=-1)  # B

    def forward(self, trajectories, cond=None, t=None):
        """
        向量化计算轨迹价值
        Input:  trajectories - [batch_size, horizon, 2] (坐标范围[0,1])
        Output: valu es - [batch_size]
        """
        # 将坐标映射到网格索引 [-1,1] -> [0, width-1]
        '''sequences = self.normalizer.unnormalize(utils.to_np(trajectories[:,:,self.action_dim:]), 'observations')
        grid_coords = torch.from_numpy(sequences[:,:,:self.state_dim]).to(self.device)'''
        #grid_coords = trajectories[:,:,self.action_dim:].to(self.device)
        sequences = self.normalizer.unnormalize(trajectories[:,:,self.action_dim:], 'observations')
        grid_coords = sequences[:,:,:self.state_dim].to(self.device)
        #grid_coords = trajectories.to(self.device)
        #self.goal = grid_coords[:, -1, :].to(self.device)  # 使用轨迹的最后一个位置作为目标

        indices = torch.round(grid_coords).long().to(self.device)
        #print(indices)
        indices = indices.clip(self.bound_min, self.bound_max)
        #print(indices)
        # 计算每个位置到目标的距离

        values = self._compute_values_batch(grid_coords, indices)
        
        
        return values

    def gradients(self, x, *args):
        """计算value对输入的梯度（自动向量化）"""
        x = x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad(y.sum(), x, create_graph=False)[0]
        return y, grad


class CostGuide_maze2d(nn.Module): 
    def __init__(self, model, maze_layout=None, normalizer=None, device="cuda", loss_type='l2', neighbour_num=4, action_dim=2, state_dim=2):
        super().__init__()
        # 迷宫布局定义 
        self.maze_layout = maze_layout or \
            "############\\"+\
            "#OOOO#OOOOO#\\"+\
            "#O##O#O#O#O#\\"+\
            "#OOOOOO#OOO#\\"+\
            "#O####O###O#\\"+\
            "#OO#O#OOOOO#\\"+\
            "##O#O#O#O###\\"+\
            "#OO#OOO#OGO#\\"+\
            "############"
        
        # 解析迷宫并预计算距离场
        self.model = model
        self.device = device
        self.neighbour_num = neighbour_num
        if neighbour_num == 4:
            self.directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=self.device)
        elif neighbour_num == 8:    
            self.directions = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]], device=self.device)
        elif neighbour_num == 9:
            self.directions = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [0, 0]], device=self.device)
        else:
            raise ValueError("neighbour_num must be 4, 8 or 9")
        self.bias = 1
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        #self.distance_map = self._precompute_distance_map()
        self.normalizer = normalizer
        # 注册为buffer以支持GPU
        #self.register_buffer("distance_map_tensor", 
        #                  torch.from_numpy(self.distance_map).float())
        self.maze_grid = self._parse_maze()
        self.height, self.width = self.maze_grid.shape
        

        self.bound_min = torch.tensor([0, 0], device=self.device)
        self.bound_max = torch.tensor([self.height-1, self.width-1], device =self.device)
        self._precompute_neighbours()
        self.to(device)
        if loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError("loss_type must be 'l2' or 'l1'")
    def _parse_maze(self):
        rows = self.maze_layout.split('\\')
        maze_np = np.array([[1 if c == '#' else 0 for c in row] for row in rows])
        return torch.from_numpy(maze_np).to(self.device)  
 

    def _precompute_neighbours(self):
        neighbours = torch.zeros((self.height, self.width, self.neighbour_num), dtype=torch.bool, device=self.device)

        for i in range(self.height):
            for j in range(self.width):
                neighbour = (torch.tensor([i, j], device = self.device)+self.directions).clip(self.bound_min, self.bound_max)
                #print(neighbours.shape)
                neighbours[i, j] = self.maze_grid[neighbour[:, 0], neighbour[:, 1]]

        self.neighbours = neighbours.to(self.device)
 

    def forward(self, x):
        """
        向量化计算轨迹价值
        Input:  trajectories - [batch_size, horizon, 4+4(grad)+neighbour] (坐标范围[0,1])
        Output: values - [batch_size, horizon, 1]]
        """
        values = self.model(x)
        return values  

        

    def _preprocess_trajectories(self, trajectories):
        # preprocess trajectories to match the format for the model input
        # trajectories [b, H, 6] -> [b*(H-1), 4(current state) + 4(next_relatice_state) + self.neighbour_num]
        # model output [b*(H-1), 1]
        # value [b, 1] one velue for each trajectory
        #for train
        sequences = self.normalizer.unnormalize(trajectories[:,:,self.action_dim:], 'observations')
        #for test
        #print("trajectories shape:", trajectories.shape)
        #sequences = trajectories[:,:,self.action_dim:]
        #print(sequences.shape)
        grid_coords = sequences[:,:,:self.state_dim].to(self.device)
        #print(grid_coords.shape)
        
        
        # 四舍五入并转为整数索引 (需要clamp防止越界)
        indices = torch.round(grid_coords).long().to(self.device)
        #print(indices)
        indices = indices.clip(self.bound_min, self.bound_max).detach()
        current_indices = indices[:, :-1, :].detach()  # 去掉最后一个位置的索引
        #print(indices)


        current_relative_grid_coords = (grid_coords[:, :-1, :] - current_indices.float()).detach()  # 去掉最后一个位置
        next_relative_grid_coords = grid_coords[:, 1:, :] - current_indices.float()  # 去掉第一个位置

        #relative_grid_coords = (grid_coords - indices)

        #current_relative_grid_coords = relative_grid_coords[:, :-1, :].detach()  # 去掉最后一个位置
        #next_relative_grid_coords = relative_grid_coords[:, 1:, :] # 去掉第一个位置
        current_velocity = sequences[:, :-1, self.state_dim:].to(self.device).detach()  # 去掉最后一个位置的速度
        next_velocity = sequences[:, 1:, self.state_dim:].to(self.device) # 去掉第一个位置的速度
        current_neighbours = self.neighbours[current_indices[:, :, 0], current_indices[:, :, 1]].float().detach()  # 去掉最后一个位置的邻居
        

        out =  torch.cat((current_relative_grid_coords, current_velocity, next_relative_grid_coords, next_velocity, current_neighbours), dim=-1).to(self.device)
        
        

        # 拼接起来
        return  out


    def gradients(self, trajectories, *args):
        """计算value对输入的梯度（自动向量化）"""
        #x is trajectories [batch, horizon, 2 action_dim+ 2 position_dim + 2 velocity_dim] normalized

        trajectories = trajectories.requires_grad_()
        x = self._preprocess_trajectories(trajectories)
        
        y = self(x).sum(dim=[-1,-2])  # [b,
        
        grad = torch.autograd.grad(y.mean(), trajectories)[0]
        
        return y, grad

    def get_training_data(self, observation, next_observation, next_waypoint):
        """
            处理获得输进网络的数据 （x = 4+4+neighbour_num, y = 1）
        """
        observation = torch.from_numpy(observation)
        next_observation = torch.from_numpy(next_observation)
        next_waypoint = torch.from_numpy(next_waypoint)
        
        grid_coord = observation[:self.state_dim]
        indice = torch.round(grid_coord)
        indice = indice.clip(self.bound_min.to('cpu'), self.bound_max.to('cpu')).long()

        x = torch.cat((
            grid_coord - indice.float(),
            observation[self.state_dim:],
            next_waypoint[:self.state_dim] - indice.float(),
            next_waypoint[self.state_dim:],
            self.neighbours[indice[0], indice[1]].float().to('cpu'),
        ))

        y = -torch.norm(next_observation - next_waypoint)
        return x, y  # 计算下一个位置到下一个目标点的距离

    def loss(self, x, y, *args):
        """
        计算损失函数
        Input:  trajectories - [batch_size, horizon, 2 action_dim+ 2 position_dim + 2 velocity_dim] normalized
        y - [batch_size, horizon, 1] (ground truth values)
        Output: loss - scalar
        """
        pred_y = self(x, *args)
        loss = self.loss_fn(pred_y, y)
        return loss

class CostPositionGuide_maze2d(CostGuide_maze2d):  #execution based only position, w/o velocity
    def __init__(self, model, maze_layout=None, normalizer=None, device="cuda", loss_type='l2', neighbour_num=4, action_dim=2, state_dim=2):
        super().__init__(model = model, maze_layout=maze_layout, normalizer=normalizer, device=device, loss_type=loss_type, neighbour_num=neighbour_num, action_dim=action_dim, state_dim=state_dim)
        # 迷宫布局定义 
       
    

        

    def _preprocess_trajectories(self, trajectories):
        # preprocess trajectories to match the format for the model input
        # trajectories [b, H, 6] -> [b*(H-1), 4(current state) + 4(next_relatice_state) + self.neighbour_num]
        # model output [b*(H-1), 1]
        # value [b, 1] one velue for each trajectory
        #for train
        sequences = self.normalizer.unnormalize(trajectories[:,:,self.action_dim:], 'observations')
        #for test
        #print("trajectories shape:", trajectories.shape)
        #sequences = trajectories[:,:,self.action_dim:]
        #print(sequences.shape)
        grid_coords = sequences[:,:,:self.state_dim].to(self.device)
        #print(grid_coords.shape)
        
        
        # 四舍五入并转为整数索引 (需要clamp防止越界)
        indices = torch.round(grid_coords).long().to(self.device)
        #print(indices)
        indices = indices.clip(self.bound_min, self.bound_max).detach()
        current_indices = indices[:, :-1, :].detach()  # 去掉最后一个位置的索引
        #print(indices)


        current_relative_grid_coords = (grid_coords[:, :-1, :] - current_indices.float()).detach()  # 去掉最后一个位置
        next_relative_grid_coords = grid_coords[:, 1:, :] - current_indices.float()  # 去掉第一个位置

        #relative_grid_coords = (grid_coords - indices)

        #current_relative_grid_coords = relative_grid_coords[:, :-1, :].detach()  # 去掉最后一个位置
        #next_relative_grid_coords = relative_grid_coords[:, 1:, :] # 去掉第一个位置
        #current_velocity = sequences[:, :-1, self.state_dim:].to(self.device).detach()  # 去掉最后一个位置的速度
        #next_velocity = sequences[:, 1:, self.state_dim:].to(self.device) # 去掉第一个位置的速度
        current_neighbours = self.neighbours[current_indices[:, :, 0], current_indices[:, :, 1]].float().detach()  # 去掉最后一个位置的邻居
        

        out =  torch.cat((current_relative_grid_coords, next_relative_grid_coords, current_neighbours), dim=-1).to(self.device)
        
        

        # 拼接起来
        return  out


    def get_training_data(self, observation, next_observation, next_waypoint):
        """
            处理获得输进网络的数据 （x = 2+2+neighbour_num, y = 1）
        """
        observation = torch.from_numpy(observation)
        next_observation = torch.from_numpy(next_observation)
        next_waypoint = torch.from_numpy(next_waypoint)
        
        grid_coord = observation[:self.state_dim]
        indice = torch.round(grid_coord)
        indice = indice.clip(self.bound_min.to('cpu'), self.bound_max.to('cpu')).long()

        x = torch.cat((
            grid_coord - indice.float(),
            next_waypoint[:self.state_dim] - indice.float(),
            self.neighbours[indice[0], indice[1]].float().to('cpu'),
        ))

        y = -torch.norm(next_observation[:2] - next_waypoint[:2])
        return x, y  # 计算下一个位置到下一个目标点的距离




if __name__ == "__main__":
    # 测试 ValueGuide_maze2d_v4
    maze_layout = "############\\"+\
                  "#OOOO#OOOOO#\\"+\
                  "#O##O#O#O#O#\\"+\
                  "#OOOOOO#OOO#\\"+\
                  "#O####O###O#\\"+\
                  "#OO#O#OOOOO#\\"+\
                  "##O#O#O#O###\\"+\
                  "#OO#OOO#OGO#\\"+\
                  "############"
    model = MultiLinearLayer(input_dim=8, output_dim=1)
    #guide = ValueGuide_maze2d_v4(maze_layout=maze_layout, device="cuda")
    guide = CostPositionGuide_maze2d(model, maze_layout = maze_layout, neighbour_num = 4)


    '''x, y = guide.get_training_data(
        observation=torch.tensor([7.1, 2.1, 0.13, 0.0]),
        next_observation=torch.tensor([7.2, 2.2, 1.0, -0.2]),
        next_waypoint=torch.tensor([7.1, 2.2, -0. , -0.1])
    )
    print("x:", x)
    print("y:", y)'''
    #trajectories = torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]], device="cuda")  # 示例轨迹
    #values = guide(trajectories)
    #print(values)
    #trajectories = torch.randn(1, 5, 2).requires_grad_(True)  # [batch, horizon, 2]
    
    trajectories = torch.tensor([[[0,0,1.1, 0.1,0.01,0], [0, 0, 2.2, -0,0.01,0], [0,0,3.0, -0.2,-0.01,0], [0,0,2.3, -0.1,0.02,0], [0,0,4.1, 0.2, 0.0,0.0]]]).requires_grad_(True)  # 示例轨迹
    for i in range(10):
        y, grad = guide.gradients(trajectories)
        print("Values:", y)
        print("Gradients:", grad)
        #print("Gradients shape:", grad.shape)
        trajectories = trajectories +  1000000*grad
    
    '''optimizer = torch.optim.Adam(guide.parameters(), lr=0.01)
    x = torch.tensor([[[ 0.1000,  0.1000,  0.0100,  0.0000,  1.2000,  0.0000,  0.0100,
           0.0000,  1.0000,  1.0000,  0.0000,  1.0000],
         [ 0.2000,  0.0000,  0.0100,  0.0000,  1.0000, -0.2000, -0.0100,
           0.0000,  1.0000,  1.0000,  0.0000,  1.0000],
         [ 0.0000, -0.2000, -0.0100,  0.0000, -0.7000, -0.1000,  0.0200,
           0.0000,  1.0000,  1.0000,  0.0000,  1.0000],
         [ 0.3000, -0.1000,  0.0200,  0.0000,  2.1000,  0.2000,  0.0000,
           0.0000,  1.0000,  1.0000,  0.0000,  1.0000]]]).to('cuda').requires_grad_(True) 
    #x = torch.tensor([[[0,0,1.1, 0.1,1.0,0], [0, 0, 2.2, -0,1.0,0], [0,0,3.0, -0.2,-1.0,0], [0,0,2.3, -0.1,2.0,0], [0,0,4.1, 0.2, 0.0,0.0]]]).requires_grad_(True)  # 示例轨迹
    y = torch.tensor([[[-0.2], [-0.1], [-0.3], [-0.5]]]).to("cuda")  # 示例
    for i in range(100):
        optimizer.zero_grad()
        loss = guide.loss(x, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i+1}, Loss: {loss.item()}") '''