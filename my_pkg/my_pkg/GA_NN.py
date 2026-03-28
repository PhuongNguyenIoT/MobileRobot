import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import random
import time

class GA_NN(Node):
    def __init__(self):
        super().__init__('ga_nn_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription_scan = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.subscription_odom = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.subscription_fitness = self.create_subscription(Float32, 'fitness', self.fitness_callback, 10)
        self.subscription_generation = self.create_subscription(Int32, 'generation', self.generation_callback, 10)
        self.subscription_best_individual = self.create_subscription(String, 'best_individual', self.best_individual_callback, 10)

        # GA parameters
        self.population_size = 10
        self.num_generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

        # NN parameters
        self.input_size = 24  # Number of laser scan points
        self.hidden_size = 16
        self.output_size = 2   # Linear and angular velocity

        # Initialize GA population
        self.population = [self.create_random_individual() for _ in range(self.population_size)]
        self.current_generation = 0
        self.best_individual = None
        self.fitness_scores = np.zeros(self.population_size)
    def fitness_function(self, individual):
        # Placeholder for actual fitness evaluation logic
        # This should evaluate the individual's performance in the environment
        return random.uniform(0, 1)  # Replace with actual fitness calculation
    def create_random_individual(self):
        # Create a random individual (NN weights)
        return np.random.rand(self.input_size * self.hidden_size + self.hidden_size * self.output_size)
    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness_scores[i] = self.fitness_function(self.population[i])
    def select_parents(self):
        # Select parents based on fitness scores (roulette wheel selection)
        total_fitness = np.sum(self.fitness_scores)
        if total_fitness == 0:
            return random.choices(self.population, k=2)  # Avoid division by zero
        probabilities = self.fitness_scores / total_fitness
        return np.random.choice(self.population, size=2, p=probabilities)
    def crossover(self, parent1, parent2):        
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.1)  # Add small random noise
        return individual
    def run_ga(self):
        for generation in range(self.num_generations):
            self.evaluate_population()
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            self.population = new_population
            self.current_generation = generation
            self.best_individual = self.population[np.argmax(self.fitness_scores)]
            self.get_logger().info(f'Generation {generation}: Best Fitness = {np.max(self.fitness_scores)}')
    def scan_callback(self, msg):
        # Process laser scan data and use it as input for the NN
        pass  # Implement processing logic here
    def odom_callback(self, msg):
        # Process odometry data
        pass  # Implement processing logic here
    def fitness_callback(self, msg):
        # Update fitness scores based on received data
        pass  # Implement logic to update fitness scores here
    def generation_callback(self, msg):
        pass
