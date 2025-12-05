import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from diet4cola.model.models import UNet

DEVICE = 'cpu'

def load_unet_model(id: str) -> nn.Module:
    model = UNet(base_ch=32, in_ch=2, out_ch=2)
    model.load_state_dict(torch.load(f'../../models/model_{id}'))
    model.to(DEVICE)
    model.eval()

    return model

def get_motion_fields(frames: np.ndarray, model: nn.Module) -> np.ndarray:
    if frames.ndim != 3:
        raise ValueError("Expected three dimensions <num_frames, 512, 512>")
    if frames.shape[1] != 512 or frames.shape[2] != 512:
        raise ValueError("Expected shape of frame to be 512x512")
    
    num_frames = frames.shape[0]
    model_inputs = []
    
    for i in range(num_frames - 1):
        frame_a = torch.tensor(frames[i], dtype=torch.float32)
        frame_b = torch.tensor(frames[i + 1], dtype=torch.float32)
        input = torch.stack((frame_a, frame_b)).unsqueeze(0)
        model_inputs.append(input)

    outputs = []
    with torch.no_grad():
        for input in tqdm(model_inputs, 'Processing...'):
            output = model(input).cpu().detach().numpy().squeeze().squeeze()

            velocity_x = output[0]
            velocity_y = output[1]
            outputs.append((velocity_x, velocity_y))

    return np.stack(outputs)

def get_trajectories(points: np.ndarray, frames: np.ndarray, model: nn.Module) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("Expected three dimensions <num_points, 2>")
    if points.shape[1] != 2:
        raise ValueError("Points are expected to have two elements")

    velocity_fields = get_motion_fields(frames, model)
    num_points = points.shape[0]

    trajectories = {}
    for i in range(num_points):
        point = points[i]
        trajectories[i] = [point]

    index = 1
    for vf in tqdm(velocity_fields, desc='Tracking...'):
        vf_x = vf[0]
        vf_y = vf[1]

        for i in range(num_points):
            point = points[i]
            previous_position = (trajectories[i])[index - 1]

            ipx = int(previous_position[0])
            ipy = int(previous_position[1])

            v_x = vf_x[ipx, ipy]
            v_y = vf_y[ipx, ipy]

            next_x = previous_position[0] + v_x
            next_y = previous_position[1] + v_y

            trajectories[i].append(np.array([next_x, next_y]))