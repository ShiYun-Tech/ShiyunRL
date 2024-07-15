import gymnasium as gym
import geneticSearch
import json
import torch

# 超参数范围字典
genetics = {
    'algo':'ppo',
    'actor_lr': [3e-5, 1e-3],
    'critic_lr': [3e-5, 1e-3],
    'hidden_dim': [128, 256],
    'capacity': [2000, 5000],
    'gamma': [0.8, 0.99],
    'tau': [0.01, 0.02],
    'seed': [0, 10],
    'episodes': [2000, 2000],
    'batch_size': [64, 256]
}

if __name__ == '__main__':
    # 将 Python 字典转换成 JSON 字符串
    genetics_json = json.dumps(genetics, indent=4)

    # 写入文件
    with open('./hyper_parameter/genetics.json', 'w') as json_file:
        json_file.write(genetics_json)
    # Make the Environment
    env = gym.make('CartPole-v1')#,render_mode='human'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Read the hyper parameter from JSON
    with open('./hyper_parameter/genetics.json', 'r') as json_file:
        genetics = json.load(json_file)

    # Genetic Search part
    best_individual, best_score = geneticSearch.GeneticSearch(env = env,genetics=genetics,device=device,population_size = 20,elite_frac = 0.5,num_generations = 2).evolve()
    
    # Save the best hyper parameter as YAML file
    best_individual.save_model()

    print("Completed the Train.The Best Score is ",best_score)