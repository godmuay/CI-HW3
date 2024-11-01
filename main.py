import numpy as np
import random
import matplotlib.pyplot as plt


class FeedforwardNN:
    def __init__(self, architecture):
        self.architecture = architecture
       
        self.weights = [np.random.uniform(-1, 1, (architecture[i], architecture[i + 1])) for i in range(len(architecture) - 1)]


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        activations = input_data
        for weight in self.weights:
            activations = self.sigmoid(np.dot(activations, weight))
        return activations

    
    def calculate_loss(self, true_output, predicted_output):
        return np.mean((true_output - predicted_output) ** 2)


class GeneticOptimizer:
    def __init__(self, training_data, training_labels, nn_structure, population_size, max_generations, mutation_rate):
        self.training_data = training_data
        self.training_labels = training_labels
        self.nn_structure = nn_structure
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate


    def initialize_population(self):
        return [FeedforwardNN(self.nn_structure) for _ in range(self.population_size)]

    
    def mutate(self, weights):
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.normal(0, 0.5, weights[i].shape)
        return weights

   
    def crossover(self, parent_a, parent_b):
        child_weights = []
        for w1, w2 in zip(parent_a.weights, parent_b.weights):
            mask = np.random.rand(*w1.shape) < 0.5
            child_weights.append(np.where(mask, w1, w2))
        return child_weights

    
    def evolve_population(self):
        population = self.initialize_population()
        fitness_history = []

        for generation in range(self.max_generations):
            fitness_scores = []
            for network in population:
                predictions = np.round([network.predict(x) for x in self.training_data])
                fitness_scores.append(np.mean(predictions.flatten() == self.training_labels))

            
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices[:self.population_size // 2]]

            offspring = []
            while len(offspring) < self.population_size:
                parent_a, parent_b = random.sample(population, 2)
                child_weights = self.crossover(parent_a, parent_b)
                child_weights = self.mutate(child_weights)
                child_network = FeedforwardNN(self.nn_structure)
                child_network.weights = child_weights
                offspring.append(child_network)
            
            population.extend(offspring)
            fitness_history.append(max(fitness_scores))
            print(f"Generation {generation + 1}, Best Fitness: {max(fitness_scores)}")

        return population[0], fitness_history


def load_dataset(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=str)
    features = data[:, 2:].astype(np.float32)
    labels = np.array([1 if diagnosis == 'M' else 0 for diagnosis in data[:, 1]])
    return features, labels


def k_fold_split(data, labels, folds=10):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data_folds = np.array_split(data[indices], folds)
    label_folds = np.array_split(labels[indices], folds)
    
    for i in range(folds):
        x_test = data_folds[i]
        y_test = label_folds[i]
        x_train = np.concatenate([data_folds[j] for j in range(folds) if j != i])
        y_train = np.concatenate([label_folds[j] for j in range(folds) if j != i])
        yield x_train, y_train, x_test, y_test


def plot_confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=int).flatten(), np.array(y_pred, dtype=int).flatten()
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in [0, 1] and p in [0, 1]:
            confusion_matrix[t, p] += 1

    TP, FP, FN, TN = confusion_matrix[1, 1], confusion_matrix[0, 1], confusion_matrix[1, 0], confusion_matrix[0, 0]
    accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
    
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap='Greens') 
    plt.colorbar(ax.matshow(confusion_matrix, cmap='Greens'))

    for (i, j), value in np.ndenumerate(confusion_matrix):
        plt.text(j, i, value, ha='center', va='center', color='red')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.text(0.85, 1.60, f'Accuracy: {accuracy:.2f} %')
    plt.xticks([0, 1], ['B', 'M'])
    plt.yticks([0, 1], ['B', 'M'])
    
    return accuracy


if __name__ == '__main__':
    data_file = 'wdbc.data'
    features, labels = load_dataset(data_file)
    
    num_folds = 10
    nn_architecture = [30, 50, 1]
    population_size = 30
    generations_count = 100
    mutation_probability = 0.01

    fold_accuracies = []
    for fold_index, (train_data, train_labels, test_data, test_labels) in enumerate(k_fold_split(features, labels, num_folds), start=1):

        optimizer = GeneticOptimizer(train_data, train_labels, nn_architecture, population_size, generations_count, mutation_probability)
        best_network, fitness_progress = optimizer.evolve_population()

     
        predictions = np.array([np.round(best_network.predict(x)) for x in test_data]).flatten()
        accuracy = plot_confusion_matrix(test_labels, predictions)
        fold_accuracies.append(accuracy)

        plt.figure(0)
        plt.subplot(1, num_folds, fold_index)
        plt.grid()
        if fold_index == 1:
            plt.ylabel('Best Fitness')
        plt.xlabel(f'Fold {fold_index}')
        plt.plot(fitness_progress)
    
    plt.suptitle('Genetic Algorithm Training Progress')
    print("Fold Accuracies: {} %".format(fold_accuracies))
    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average Accuracy over {num_folds} folds: {avg_accuracy:.2f}%")
    plt.show()


