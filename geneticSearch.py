import random
import numpy as np
from algorithms.sac.runner import sacRunner
from algorithms.ppo.runner import ppoRunner

class GeneticSearch():
    # 遗传算法参数
    def __init__(self,env,genetics,device,population_size = 5,elite_frac = 0.2,num_generations = 10):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.population_size = population_size
        self.num_generations = num_generations
        self.elite_frac = elite_frac
        self.genetics = genetics
        self.elite_count = int(elite_frac * population_size)
        self.device = device
        self.population = self.create_initial_population()
        
    # 初始化种群
    def create_initial_population(self):
        population = []

        for _ in range(self.population_size):
            actor_lr = random.uniform(*self.genetics['actor_lr'])
            critic_lr = random.uniform(*self.genetics['critic_lr'])
            tau = random.uniform(*self.genetics['tau'])
            seed = random.randint(*self.genetics['seed'])
            episodes = random.randint(*self.genetics['episodes'])
            batch_size = random.randint(*self.genetics['batch_size'])
            gamma = random.uniform(*self.genetics['gamma'])
            hidden_dim = random.randint(*self.genetics['hidden_dim'])
            capacity = random.randint(*self.genetics['capacity'])
            if self.genetics['algo'] == 'sac':
                agent = sacRunner(self.state_dim,self.action_dim,hidden_dim,actor_lr,critic_lr,gamma,tau,seed,episodes,capacity,batch_size,self.device)
            elif self.genetics['algo'] == 'ppo':
                agent = ppoRunner(self.state_dim,self.action_dim,hidden_dim,actor_lr,critic_lr,gamma,tau,seed,episodes,capacity,batch_size,self.device)
            else:
                raise ValueError("error: check the genetics algo.")
            
            population.append(agent)

        return population

    # 计算适应度得分
    def calculate_population_fitness(self,population):
        scores = []

        for agent in population:
            score = agent.evaluate(self.env)
            scores.append(score)

        return scores

    # 选择精英个体
    def select_elites(self,population, scores, elite_count):
        sorted_combinations = sorted(list(zip(scores, population)), key = lambda x:x[0],reverse=True)
        sorted_population = [x[1] for x in sorted_combinations]
        elites = sorted_population[:elite_count]

        return elites

    # 选择父母
    def select_parents(self,population, scores):
        # 使用轮盘赌选择父母
        probabilities = np.array(scores) / sum(scores)
        parents = random.choices(population, probabilities, k=2)

        return parents

    # 交叉操作
    def crossover(self,parents):
        # 进行交叉操作生成后代
        actor_lr = parents[random.choice([0,1])].actor_lr
        critic_lr = parents[random.choice([0,1])].critic_lr
        batch_size = parents[random.choice([0,1])].batch_size
        gamma = parents[random.choice([0,1])].gamma
        tau = parents[random.choice([0,1])].tau
        episodes = parents[random.choice([0,1])].episodes
        hidden_dim = parents[random.choice([0,1])].hidden_dim
        capacity = parents[random.choice([0,1])].capacity
        seed = parents[random.choice([0,1])].seed

        if self.genetics['algo'] == 'sac':
            agent = sacRunner(self.state_dim,self.action_dim,hidden_dim,actor_lr,critic_lr,gamma,tau,seed,episodes,capacity,batch_size)
        elif self.genetics['algo'] == 'ppo':
            agent = ppoRunner(self.state_dim,self.action_dim,hidden_dim,actor_lr,critic_lr,gamma,tau,seed,episodes,capacity,batch_size)
        else:
            raise ValueError("error: check the genetics algo.")        
        
        return agent
        

    # 变异操作
    def mutate(self,offspring,mutation_rate = 0.05):
        # 进行变异操作生成变异后代
        if random.uniform(0,1) < mutation_rate:    offspring.actor_lr = random.uniform(*self.genetics['actor_lr'])
        if random.uniform(0,1) < mutation_rate:    offspring.critic_lr = random.uniform(*self.genetics['critic_lr'])
        if random.uniform(0,1) < mutation_rate:    offspring.tau = random.uniform(*self.genetics['tau'])
        if random.uniform(0,1) < mutation_rate:    offspring.seed = random.randint(*self.genetics['seed'])
        if random.uniform(0,1) < mutation_rate:    offspring.episodes = random.randint(*self.genetics['episodes'])
        if random.uniform(0,1) < mutation_rate:    offspring.batch_size = random.randint(*self.genetics['batch_size'])
        if random.uniform(0,1) < mutation_rate:    offspring.gamma = random.uniform(*self.genetics['gamma'])
        if random.uniform(0,1) < mutation_rate:    offspring.hidden_dim = random.randint(*self.genetics['hidden_dim'])
        if random.uniform(0,1) < mutation_rate:    offspring.capacity = random.randint(*self.genetics['capacity'])
        
        return offspring

    # 寻找适应度最高的个体
    def find_best_individual(self,population, scores):
        best_index = np.argmax(scores)
        return population[best_index], scores[best_index]

    # 遗传算法主函数
    def evolve(self):
        population = self.create_initial_population()
        elite_count = int(self.elite_frac * self.population_size)

        for i in range(self.num_generations):
            scores = self.calculate_population_fitness(population)
            elites = self.select_elites(population, scores, elite_count)
            next_generation = []
            next_generation.extend(elites)

            while len(next_generation) < self.population_size:
                parents = self.select_parents(population, scores)
                offspring = self.crossover(parents)
                offspring = self.mutate(offspring)
                next_generation.append(offspring)

            population = next_generation

        best_individual, best_score = self.find_best_individual(population, scores)

        return best_individual, best_score