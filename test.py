from algorithms.ppo.runner import ppo_gene_test_runner
from algorithms.sac.runner import sac_gene_test_runner
import gymnasium as gym
import yaml
import argparse

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r', encoding='utf-8') as file:  # 指定编码为 utf-8
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn")
    args = parser.parse_args()
    env = gym.make('CartPole-v1')#,render_mode='human'
    if args.algo == 'ppo':
        config = load_config('./configs/ppo_best_gene_config.yaml')
        ppo_gene_test_runner(env, config)
    elif args.algo == 'sac':
        config = load_config('./configs/sac_best_gene_config.yaml')
        sac_gene_test_runner(env, config)
    else:
        raise ValueError("Error: Point the right algorithm.")
