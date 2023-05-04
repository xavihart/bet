import torch
import torchvision.io as io
import json
import wandb
import os
import glob
from transformers import CLIPModel
import torch.nn as nn


class Carla_move_action_data(torch.utils.data.Dataset):
    def __init__(self, path, mode='train', train_split=0.7, ):
        super().__init__()

        
        
        dir_name = glob.glob(path+'/*/')
        # print(dir_name)
        dir_name.sort()

        dir_num = len(dir_name)

        if mode == 'train':
            self.dir_name = dir_name[:int(dir_num*train_split)]
        elif mode == 'val':
            self.dir_name = dir_name[int(dir_num*train_split):]
        
        self.len = len(self.dir_name)



    def load_obs(self, p):
        video, _, _ = io.read_video(p)
        video = video.permute(0, 3, 1, 2) # frame_num * 3 * h *w
        video = video / 255
        return video
    

    def load_action(self, p):
        # load action: dim=2
        action = json.load(open(p, 'rb'))
        action_len = len(action)
        action_seq = torch.zeros(action_len, 2)
        for i in range(action_len):
            action_seq[i][0] = action[i]['throttle']
            action_seq[i][1] = action[i]['steer']
        return action_seq

        
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        p = self.dir_name[i]
        p_obs, p_act = os.path.join(p, '0.mp4'), os.path.join(p, 'actions.json')
        obs, act = self.load_obs(p_obs), self.load_action(p_act)
        # print(obs.shape, act.shape)
        # assert obs.shape[0] == act.shape[0]
        act = act[(act.shape[0] - obs.shape[0]):, ...]
        assert obs.shape[0] == act.shape[0]
        return obs, act 





class SkillAutoEncoder(torch.nn.Module):
    def __init__(self, conf):

        self.action_dim = conf['act_dim']
        self.act_emb = conf['act_emb']
        self.obs_emb = conf['obs_emb']
        self.input_emb = conf['input_emb']
        
        self.encoder_layer_num = conf['layer_num']
        self.encoder_dim = conf['encoder_d']
        self.num_head = conf['num_head']


        self.codebook_dim = conf['codebook_dim']
        self.codebook_entry = conf['codebook_entry']

        self.skill_block_size = conf['skill_block_size']

        
        

        # define modules
        self.action_mlp = nn.Linear(self.action_dim, self.act_emb)
        self.obs_action_mlp = nn.Linear(self.act_emb + self.obs_emb, self.input_emb)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_emb, nhead=self.num_head, batch_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_layer_num)

        self.codebook = nn.Embedding(self.codebook_entry, self.codebook_dim)
        self.prediction_head = nn.Linear(self.input_emb * self.skill_block_size, self.codebook_entry)


        self.skill_mlp = nn.Sequantial([
                           nn.Linear(self.codebook_dim + self.obs_emb, self.input_emb * 2),
                           nn.ReLU(),
                           nn.Linear(self.input_emb * 2, self.input_emb * 2),
                           nn.ReLU(),
                           nn.Linear(self.input_emb * 2, self.input_emb * 2),
                           nn.Linear(self.input_emb * 2, self.skill_block_size * self.action_dim)
        ])

    

        if conf['image_encoder'] == 'clip':
            class clip_vision_wrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    return self.model.get_image_features(x)
            
            self.image_encoder = clip_vision_wrapper(
                                    CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                                  )
        else:
            raise NotImplementedError
        
    

    def encode(self, obs, act):
        # obs: B * T * 3 * 224 * 224
        # act: B * T * 2

        B, T, C, H, W = obs.shape[0], obs.shape[1]
        obs = obs.view(B*T, C, H, W)
        obs_embed = self.image_encoder(obs)
        obs_embed = obs_embed.view(B, T, -1)
        act_embed = self.action_mlp(act)

        input_embed = torch.cat([obs_embed, act_embed], -1) # B * T * (obs+act)

        input_embed = self.obs_action_mlp(input_embed)

        out = self.encoder(input_embed)

    
    
    def decoder(self):
        pass
    
    
    def get_decode_loss(self, ):
        pass
    
    def get_codebook_KLloss(self, ):
        pass
    
    def forward(self, ):
        pass
    
    


dataset = Carla_move_action_data('data/bet_data_release/carla', 'val')
for i in range(dataset.__len__()):
    print(dataset[i])