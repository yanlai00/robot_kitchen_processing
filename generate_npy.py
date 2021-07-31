import os
from collections import defaultdict
from PIL import Image
from torchvision import transforms
import pickle
import numpy as np
import json
from pathlib import Path

path_data = '/home/asap7772/asap7772/real_data_kitchen/bridge_data'
fd = os.listdir(path_data)
fd = [os.path.join(path_data, x) for x in fd]


def deldir(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)

def center_crop(path): #center crop from 480x640 to 480x480 and then resized to 64x64 and flattened as a tensor
    im = Image.open(path)

    width, height = im.size   # Get dimensions
    new_width, new_height = 480,480

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im = im.resize((64,64), Image.ANTIALIAS)
    trans = transforms.ToTensor()

    return trans(im).numpy().flatten()

def squash(path): # squash from 480x640 to 64x64 and flattened as a tensor
    im = Image.open(path)

    im = im.resize((64,64), Image.ANTIALIAS)
    trans = transforms.ToTensor()

    return trans(im).numpy().flatten()

def process_images(path): #processes images at a trajectory level
    names = sorted([x for x in os.listdir(path) if 'images' in x], key = lambda x: int(x.split('images')[1]))
    image_path = [os.path.join(path, x) for x in os.listdir(path) if 'images' in x]
    image_path = sorted(image_path, key = lambda x: int(x.split('images')[1]))

    images_out = defaultdict(list)
    
    get_num = lambda x: int(x.split('_')[1].split('.')[0])

    for i in range(len(names)):
        for im_name in sorted(os.listdir(image_path[i]), key=get_num):
            images_out[names[i]].append(squash(os.path.join(image_path[i], im_name)))

    images_out = dict(images_out)

    obs, next_obs = dict(), dict()

    for n in names:
        obs[n] = images_out[n][:-1]
        next_obs[n] = images_out[n][1:]
    return obs, next_obs

def process_state(path):
    fp = os.path.join(path, 'obs_dict.pkl')
    x = pickle.load(open(fp, 'rb'))  
    return x['full_state'][:-1], x['full_state'][1:]

def process_actions(path): # gets actions
    fp = os.path.join(path, 'policy_out.pkl')
    act_list = pickle.load(open(fp, 'rb'))
    return [x['actions'] for x in act_list]    

def process_dc(path): #processes each data collection attempt
    # print(path)
    if path[-4:] == 'lmdb':
        return []

    all_dicts = list()

    # Data collected prior to 7-23 has a delay of 2, otherwise a delay of 1
    try:
        metadata_dir = os.path.join(path, 'collection_metadata.json')
    except FileNotFoundError as e:
        print(e)
        deldir(path)
        return []
    metadata_dict = json.load(open(metadata_dir))
    camera_variation = metadata_dict['camera_variation']
    date = camera_variation.split('_')[0]
    month, day = date.split('-')
    month, day = int(month), int(day)
    print(month, day)
    if month > 7 or (month == 7 and day >= 23):
        latency_shift = 0
    else:
        latency_shift = 1

    try:
        for t in sorted(os.listdir(os.path.join(path, 'raw', 'traj_group0')), key = lambda x: int(x.split('traj')[1])):
            out = dict()
            tp = os.path.join(path, 'raw', 'traj_group0', t)
            
            ld = os.listdir(tp)

            assert 'obs_dict.pkl' in ld,  tp + ':' + str(ld) 
            assert 'policy_out.pkl' in  ld,  tp + ':' + str(ld)
            assert 'agent_data.pkl' in  ld,  tp + ':' + str(ld)
            
            obs, next_obs = process_images(tp)
            acts = process_actions(tp)
            state, next_state = process_state(tp)
            term = [0] * len(acts)
            
            out['observations'] = obs
            out['observations']['state'] = state
            out['next_observations'] = next_obs
            out['next_observations']['state'] = next_state

            out['observations'] = [dict(zip(out['observations'],t)) for t in zip(*out['observations'].values())]
            out['next_observations'] = [dict(zip(out['next_observations'],t)) for t in zip(*out['next_observations'].values())]
            
            out['actions'] = acts
            # shift the actions according to camera latency
            if latency_shift:
                out['actions'] = np.concatenate([out['actions'][:1]] * latency_shift + [out['actions'][:-latency_shift]])

            out['terminals'] = term
            
            all_dicts.append(out)
    
    except FileNotFoundError as e:
        print(e)
        deldir(path)
        return []

    return all_dicts

def make_numpy(path, outpath): # overarching npy creation
    Path(outpath).mkdir(parents=True, exist_ok=True)
    lst = list()
    print(path)
    for dc in os.listdir(path):
        curr = process_dc(os.path.join(path,dc))
        lst.extend(curr)
    np.save(os.path.join(outpath, 'out.npy'), lst)
    print('saved', os.path.join(outpath, 'out.npy'))

if __name__ == "__main__":
    target_path = '/home/yanlaiyang/real_data_kitchen/'
    conv_string = lambda x: target_path + 'bridge_data_numpy' + x.split('bridge_data')[1]
    for p in fd:
        for t in os.listdir(p):
            tp = os.path.join(p,t)
            make_numpy(tp, conv_string(tp))