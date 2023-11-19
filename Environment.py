import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import torch
import random
from h5_reader import ReadH5d
from training_helpers import dice_metric
# from stable_baselines3 import DDPG




class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, model, data_dir_list, H_a, W_a, D_a, H_o, W_o, D_o, device='cpu'):
        super().__init__()
        #####################
        # model and data_dir_list is the env
        self.device = device
        model.eval()
        model.to(device)
        self.model = model
        self.data_dir_list = data_dir_list
        self.data_loader = ReadH5d()
        ######################
        self.action_space = spaces.Box(low=0, high=1,
                                            shape=(H_a, W_a, D_a, 3), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, 1, H_o, W_o, D_o), dtype=np.float32)
        self.state = None

    def step(self, action):
        # action is shape (H_a, W_a, D_a, 3)
        action = torch.tensor(action).unsqueeze(0).to(device)
        # observation
        new_data_dir = random.sample(self.data_dir_list, 1)[0]
        new_data = self.data_loader(new_data_dir)
        new_data = torch.stack([new_data['image'], new_data['label']]).numpy()
        # print(data)
        # return observation, info
        
        # reward
        # read the data and send to device
        data = self.state
        img, label = data
        img, label = torch.tensor(img).unsqueeze(0).to(self.device), torch.tensor(label).unsqueeze(0).to(self.device)
        
        
        
        # reward
        processed_img, processed_label = torch.nn.functional.grid_sample(img, action), torch.nn.functional.grid_sample(label, action)
        output = self.model(processed_img)
        reward = dice_metric(output, processed_label).item()
        
        terminated = False
        
        truncated = False
        # tranistion to new state
        self.state = new_data


        return self.state, reward, terminated, truncated, {}


    def reset(self, seed=0):
        # load the file location of a random sampled data
        data_dir = random.sample(self.data_dir_list, 1)[0]
        # print(data_dir)
        # read the data and send to device
        data = self.data_loader(data_dir)
        data = torch.stack([data['image'], data['label']]).numpy()
        # return observation, info
        self.state = data
        return self.state, {'data_dir': data_dir}



if __name__ == "__main__":
    from data_dir_shuffle import read_data_list
    from monai.networks.nets.swin_unetr import SwinUNETR
    
    device = 'cpu'
    model = SwinUNETR((128, 128, 64), 1, 1)
    model.load_state_dict(torch.load('/home/xiangcen/AutoPre/seg_results/seg_model_only_batch3.ptm', map_location=device))
    model.to(device)
    
    
    
    data_list = read_data_list('/home/xiangcen/AutoPre/MSD_prostate_h5.txt')
    
    
    my_env = CustomEnv(model, data_list, 128, 128, 128, 180, 180, 20, device=device)
    data, info = my_env.reset()
    print(data.shape, info)
    action = np.array(torch.rand(128, 128, 64, 3))
    obv, reward, terminated, truncated, _ = my_env.step(action)
    print(obv.shape, reward)
    # check_env(my_env)
