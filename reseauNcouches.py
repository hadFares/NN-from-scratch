import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# n vecteur de taille égale au  nombre de couche qui
# a dans chaque composante de nombre de neurones dans la couche i.
def initialisation(dimensions) :

    nb_couches = len(dimensions) - 1 
    parametres = {} # on met tous nos parametres ans un dictionnaire
    for i in range(1, nb_couches+1) :
        parametres['W' + str(i)] = 0.1 * np.random.rand(dimensions[i], dimensions[i-1]) - 0.05
        parametres['b' + str(i)] = 0.1 * np.random.rand(dimensions[i], 1) - 0.05


    return parametres


def initialisation_Xavier(dimensions):
    parametres = {}
    for i in range(1, len(dimensions)):
        n_in = dimensions[i-1]   # Neurones de la couche précédente
        n_out = dimensions[i]    # Neurones de la couche actuelle
        
        # Initialisation Xavier/Glorot adaptée pour la sigmoïde
        variance = 2.0 / (n_in + n_out)
        parametres['W' + str(i)] = np.random.randn(n_out, n_in) * np.sqrt(variance)
        
        # Initialisation des biais à zéro (standard)
        parametres['b' + str(i)] = np.zeros((n_out, 1))
    
    return parametres


def forward_propagation(X, parametres) :

    nb_couches = len(parametres) // 2

    activations = {
        'A0' : X
    }

    for i in range(1, nb_couches+1) :
        Z = np.dot(parametres["W" + str(i)], activations['A' + str(i-1)]) + parametres["b" + str(i)]
        activations['A' + str(i)] = 1 / (1 + np.exp(-Z))


    return activations




def back_propagation(y, parametres, activations):

  m = y.shape[1]
  C = len(parametres) // 2

  dZ = activations['A' + str(C)] - y
  gradients = {}

  for c in reversed(range(1, C + 1)):
    gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
    gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    if c > 1:
      dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

  return gradients




def update(gradients, parametres, learning_rate):

    nb_couches = len(parametres) // 2
    for i in range(nb_couches) :
        parametres['W'+str(i+1)] = parametres['W'+str(i+1)] - learning_rate * gradients['dW'+str(i+1)]
        parametres['b'+str(i+1)] = parametres['b'+str(i+1)] - learning_rate * gradients['db'+str(i+1)]

    return parametres


def Loss(A, y) :
    eps = 1e-15
    return 1/y.shape[1] * np.sum(-y * np.log(A + eps) - (1 - y) * np.log(1 - A + eps))


def predict(X, parametres):
  nb_couches = len(parametres) // 2
  activations = forward_propagation(X, parametres)
  A = activations['A'+ str(nb_couches)]
  return A >= 0.5


def Train(X, y, dimensions, coef, iter) :
    parametres = initialisation_Xavier(dimensions)
    nb_couches = len(dimensions) - 1
    taille = iter // 10 + 1
    loss = [0] * taille
    k = 0

    #boucle d'apprentissage 
    for i in tqdm( range(iter) ):
        activations = forward_propagation(X, parametres)
        gradients = back_propagation(y, parametres, activations)
        parametres = update(gradients, parametres, coef)
        if i%10 == 0 :
            loss[k] = Loss(activations["A" + str(nb_couches)], y)
            k = k + 1
            

    # Prédictions après entraînement
    predictions = forward_propagation(X, parametres)
    predictions_binaires = (predictions["A"+str(nb_couches)] > 0.5).astype(int)  # Seuil à 0.5

    # Calcul de l'accuracy
    accuracy = np.mean(predictions_binaires == y)
    print(accuracy)

    #Affichage de l'évolution de la fonction cout
    plt.plot(loss)
    plt.show()

    return(parametres)