import numpy as np
import matplotlib.pyplot as plt

def process_every(data):
    rewards = []
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.tight_layout()
    plt.axis('off')
    for i in range(len(data)):
        rewards_arr = data[i]['terminals']
        print(type(rewards_arr))
        for j in range(len(rewards_arr)):
            obs = data[i]['observations'][j]
            image = obs['image_observation'].reshape(3,64,64)
            tp = image.transpose(2,1,0)
            
            plt.imshow(tp)
            plt.title('Trajectory ' + str(i) + ', Transition ' + str(j))
            plt.show()
            
            response=input()
            rewards_arr = np.array(data[i]['rewards'])
            if response != 'n':
                rewards_arr[j] = 1
            else:
                rewards_arr[j] = 0
        rewards.append(rewards_arr)

def cond_pot(data):
    gripper = np.array([x[-1] for x in data['actions']])
    for j in range(gripper.shape[-1],0,-1):
        if gripper[j-1] < 0:
            return j+1
    return -1

def cond_rev(data):
    gripper = np.array([x[-1] for x in data['actions']])
    for j in range(0, gripper.shape[-1]):
        if gripper[j] < 0: 
            return j
    return -1    

def process(data, cond_view=cond_pot, htresh=0.1, nav_type=None, rev_cond=cond_rev):
    rewards = []
    
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.tight_layout()
    plt.axis('off')
    
    for i in range(len(data)):
        if nav_type == 'even' and i % 2 != 0:
            flag=True
        elif nav_type == 'odd' and i % 2 == 0:
            flag = True
        else:
            flag = False
        
        func_cond = rev_cond if flag else cond_view
        index = func_cond(data[i])
        if index >= 0:
            num = len(data[i]['observations'])-index
            a = 1
            for j in range(index, len(data[i]['observations'])):
                obs = data[i]['observations'][j]
                # print(obs['state_observation'][-1])
                if obs['state_observation'][-1] > htresh:
                    index = j
                    image = obs['image_observation'].reshape(3,64,64)
                    break
                
            tp = image.transpose(2,1,0)
            plt.imshow(tp)
            plt.show()

            response=input()
            rewards_arr = np.array(data[i]['terminals'])
            if response != 'n':
                if flag:
                    rewards_arr[:index+1] = 1
                else:    
                    rewards_arr[index:] = 1
            else:
                print('denied')
            rewards.append(rewards_arr)
        else:
            print('invalid index')
            rewards_arr = data[i]['rewards']
            rewards.append(rewards_arr)
    return rewards

def process_new(data, cond_view=cond_pot, htresh=0, nav_type=None, rev_cond=cond_rev):
    rewards = []
    
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.tight_layout()
    plt.axis('off')
    
    for i in range(len(data)):
        if nav_type == 'even' and i % 2 != 0:
            flag=True
        elif nav_type == 'odd' and i % 2 == 0:
            flag = True
        else:
            flag = False
        
        func_cond = rev_cond if flag else cond_view
        index = func_cond(data[i])
        if index >= 0:
            num = len(data[i]['observations'])-index
            a = 1
            for j in range(index, len(data[i]['observations'])):
                obs = data[i]['observations'][j]
                # print(obs['state_observation'][-1])
                if obs['state_observation'][-1] > htresh:
                    index = j
                    image = obs['image_observation'].reshape(3,64,64)
                    break
                
            tp = image.transpose(2,1,0)
            plt.imshow(tp)
            plt.show()

            response=input()
            rewards_arr = np.array(data[i]['terminals'])
            if response != 'n':
                if flag:
                    rewards_arr[:index+1] = 1
                else:    
                    rewards_arr[index:] = 1
            else:
                print('denied')
            rewards.append(rewards_arr)
        else:
            print('invalid index')
            rewards_arr = data[i]['rewards']
            rewards.append(rewards_arr)
    return rewards
