# -*- coding: utf-8 -*-
"""
@author: Armin Khayyer
"""
import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
import zipfile
import string
import pandas as pd
from warnings import simplefilter
import os
import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot as plt
from warnings import simplefilter


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
def KerasNN(mask, X_train = X_train, y_train= y_train, X_test = X_test, y_test = y_test ):

    X_train = X_train[:10000, :, :]
    y_train = y_train[:10000]
    X_test = X_test[:1000, :, :]
    y_test = y_test[:1000]

    #masking data
    mask  = np.array(mask).reshape(28,28)
    X_train = np.multiply(X_train,mask)
    X_test = np.multiply(X_test, mask)

    # Pre-processing data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # Define Model Architecture
    NeuralNetwork = tf.keras.Sequential()
    NeuralNetwork.add(tf.keras.layers.Flatten())
    NeuralNetwork.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    NeuralNetwork.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    NeuralNetwork.add(tf.keras.layers.Dense(64, activation=tf.nn.sigmoid))
    NeuralNetwork.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # Train Model
    NeuralNetwork.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    NeuralNetwork.fit(X_train, y_train, epochs=10, batch_size = 100, verbose=0)

    # Test Model
    y_pred = NeuralNetwork.predict(X_train)
    val_loss, val_acc = NeuralNetwork.evaluate(X_test, y_test)
    return val_acc




class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness = 0
        self.chromosome_length = specified_chromosome_length
        
    def randomly_generate(self):
        for i in range(self.chromosome_length):
            if i%28 <= 5 or i%28 >= 23 :
                self.chromosome.append(random.choice([True, False, False, False, False]))
            elif i <= 5*28 or i >= 23 *28:
                self.chromosome.append(random.choice([True, False, False, False, False]))
            else:
                self.chromosome.append(random.choice([True, False, True]))
    
    def calculate_fitness(self):
        self.fitness = KerasNN(self.chromosome)



    def print_individual(self, i):
        print("Chromosome - "+str(i) +"- number of features: " + str(sum(self.chromosome)) + " Fitness: " + str(self.fitness))

class aSimpleExploratoryAttacker:
    def __init__(self, population_size, chromosome_length, mutation_rate):
        if (population_size < 2):
            print("Error: Population Size must be greater than 2")
            sys.exit()
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_amt = mutation_rate
        self.population = []
        
    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length)
            individual.randomly_generate()
            individual.calculate_fitness()
            self.population.append(individual)

    def get_worst_fit_individual(self):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for i in range(self.population_size):
            if (self.population[i].fitness < worst_fitness):
                worst_fitness = self.population[i].fitness
                worst_individual = i
            elif (self.population[i].fitness == worst_fitness):
                if sum(self.population[i].chromosome) > sum(self.population[worst_individual].chromosome):
                    worst_fitness = self.population[i].fitness
                    worst_individual = i
        return worst_individual



    # def get_worst_fit_individual(self):
    #     worst_fitness = 999999999.0  # For Maximization
    #     worst_individual = -1
    #     for i in range(self.population_size):
    #         if (self.population[i].fitness < worst_fitness):
    #             worst_fitness = self.population[i].fitness
    #             worst_individual = i
    #     return worst_individual
    
    def get_best_fitness(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return best_fitness


    # def tournoment_selection(self, k=2):
    #     tournoment_output1 = random.choices(self.population, k=k)
    #     best_indivisual1 = [i.fitness for i in tournoment_output1]
    #     parent1 = tournoment_output1[best_indivisual1.index(max(best_indivisual1))]
    #     tournoment_output2 = random.choices(self.population, k=k)
    #     best_indivisual2 = [i.fitness for i in tournoment_output2]
    #     parent2 = tournoment_output2[best_indivisual2.index(max(best_indivisual2))]
    #     return parent1, parent2



    def tournoment_selection(self, k=2):
        tournoment_output1 = random.choices(self.population, k=k)
        best_indivisual1 = [i.fitness for i in tournoment_output1]
        if best_indivisual1[0] > best_indivisual1[1]:
            parent1 = tournoment_output1[0]
        elif best_indivisual1[0] == best_indivisual1[1]:
            if sum(tournoment_output1[0].chromosome) < sum(tournoment_output1[1].chromosome):
                parent1 = tournoment_output1[0]
            else:
                parent1 = tournoment_output1[1]
        else:
            parent1 = tournoment_output1[1]

        #parent1 = tournoment_output1[best_indivisual1.index(max(best_indivisual1))]

        tournoment_output2 = random.choices(self.population, k=k)
        best_indivisual2 = [i.fitness for i in tournoment_output2]
        if best_indivisual2[0] > best_indivisual2[1]:
            parent2 = tournoment_output2[0]
        elif best_indivisual2[0] == best_indivisual2[1]:
            if sum(tournoment_output2[0].chromosome) < sum(tournoment_output2[1].chromosome):
                parent2 = tournoment_output2[0]
            else:
                parent2 = tournoment_output2[1]
        else:
            parent2 = tournoment_output2[1]
        # parent2 = tournoment_output2[best_indivisual2.index(max(best_indivisual2))]
        return parent1, parent2



    def Crossover_operator(self, mom, dad):
        kid = anIndividual(self.chromosome_length)
        kid.randomly_generate()
        for j in range(self.chromosome_length):
            prob = random.uniform(0, 1)
            prob_mut = random.uniform(0, 1)
            if prob <= .5:
                kid.chromosome[j] = mom.chromosome[j]
            else:
                kid.chromosome[j] = dad.chromosome[j]

            if prob_mut >= self.mutation_amt:
                pass
            else:
                kid.chromosome[j] = not kid.chromosome[j]
        return kid


        
    def evolutionary_cycle(self):
        mom, dad = self.tournoment_selection()
        worst_individual = self.get_worst_fit_individual()
        self.population.pop(worst_individual)
        kid = self.Crossover_operator(mom, dad)
        self.population.append(kid)
        kid.calculate_fitness()

       
    def print_population(self):
        for i in range(self.population_size):
            self.population[i].print_individual(i)
    
    def print_best_max_fitness(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        print("Best Indvidual: ",str(best_individual)," ", self.population[best_individual].chromosome, " Fitness: ", str(best_fitness))
        return self.population[best_individual]

    # def plot_evolved_candidate_solutions(self):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1,1,1,projection='3d')
    #     ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
    #     plt.title("Evolved Candidate Solutions")
    #     ax1.set_xlim3d(-100.0,100.0)
    #     ax1.set_ylim3d(-100.0,100.0)
    #     ax1.set_zlim3d(0.2,1.0)
    #     plt.show()


all_pixels = []
for rep in range(10):
    ChromLength = 28*28
    MaxEvaluations = 100

    PopSize = 20
    mu_amt  = 0.01

    simple_exploratory_attacker = aSimpleExploratoryAttacker(chromosome_length=ChromLength, mutation_rate=mu_amt, population_size=PopSize)

    simple_exploratory_attacker.generate_initial_population()
    simple_exploratory_attacker.print_population()
    best = 0
    for i in range(MaxEvaluations-PopSize):
        best = i
        simple_exploratory_attacker.evolutionary_cycle()
        if (i % PopSize == 0):
            print("At Iteration: " + str(i))
            simple_exploratory_attacker.print_population()

    print("\nFinal Population\n")
    simple_exploratory_attacker.print_population()
    best_indiv = simple_exploratory_attacker.print_best_max_fitness()
    print("Function Evaluations: " + str(i))
    # simple_exploratory_attacker.plot_evolved_candidate_solutions()
    row = best_indiv.chromosome
    print(row)
    row = np.array(row).reshape(28,28).astype("int")
    np.save( "mask.npy", row)
    plt.imshow(row)
    plt.show()
    all_pixels.append(row)

all_pixels = np.mean(all_pixels, axis=0)
import seaborn as sns
sns.heatmap(all_pixels, annot=True,  linewidths=.5, annot_kws={"size": 6})
plt.show()


