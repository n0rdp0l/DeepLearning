#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
print(tf.__version__)
print(keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# # Task 1

# ## Initial Experiments

# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# The following code cells include a loop, which goes through diffrent hyperparameters.
# Inside the loop there is a simple MLP and CNN from the book Hands on Machine Learning.
# With each iteration a diffrent hyperparameter is applied to the parameter of interest.
# Afterward the accuracies of the models with the diffrent hyperparameters are plotted in the cell after that.

# ### Initializations 

# In[ ]:


parameter =['random_normal','random_uniform','ones','glorot_uniform']
models_mlp ={}
models_cnn ={}
histories_mlp ={}
histories_cnn ={}


for i in range(len(parameter)):
    # MLP
    models_mlp["model_mlp_" + str(i)] = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(300, activation='relu',kernel_initializer=parameter[i]),
        keras.layers.Dense(100, activation='relu',kernel_initializer=parameter[i]),
        keras.layers.Dense(10, activation='softmax',kernel_initializer=parameter[i])
    ])

    models_mlp["model_mlp_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                    optimizer='sgd',
                    metrics=['accuracy'])

    histories_mlp["history_mlp_" + str(i)] = models_mlp["model_mlp_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))

    # CNN
    models_cnn["model_cnn_" + str(i)] = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='same', 
                        input_shape=[28,28,1],kernel_initializer=parameter[i]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same',kernel_initializer=parameter[i]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same',kernel_initializer=parameter[i]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu',kernel_initializer=parameter[i]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu',kernel_initializer=parameter[i]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax',kernel_initializer=parameter[i])
    ])

    models_cnn["model_cnn_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    histories_cnn["history_cnn_" + str(i)] = models_cnn["model_cnn_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))


# In[ ]:


# plot above histories of both models CNN and MLP for each of the hyperparameters
epoch = 20
fig_0, ax_0 = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
epochs = np.arange(1,epoch+1,1)
ax_0[0].set_xlabel('epochs', fontsize=20)
ax_0[0].set_ylabel('validation accuracy', fontsize=20)
ax_0[1].set_xlabel('epochs', fontsize=20)
ax_0[1].set_ylabel('validation accuracy', fontsize=20)
ax_0[0].tick_params(labelsize=15)
ax_0[1].tick_params(labelsize=15)
for i in range(len(parameter)):
    index = i
    val_accuracy_cnn = histories_cnn["history_cnn_"+str(index)].history['val_accuracy']
    val_accuracy_mlp = histories_mlp["history_mlp_"+str(index)].history['val_accuracy']
    ax_0[0].plot(epochs, val_accuracy_cnn, label=parameter[i], linewidth=4)
    ax_0[1].plot(epochs, val_accuracy_mlp, label=parameter[i], linewidth=4)
    ax_0[0].set_title("Initialization - CNN")
    ax_0[1].set_title("Initialization - MLP")
ax_0[0].legend(fontsize=20)
ax_0[1].legend(fontsize=20)
ax_0[0].grid()
ax_0[1].grid()  


# ### Activations

# In[ ]:


parameter =['relu','tanh','softmax','softsign']
models_mlp ={}
models_cnn ={}
histories_mlp ={}
histories_cnn ={}


for i in range(len(parameter)):
    models_mlp["model_mlp_" + str(i)] = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(300, activation=parameter[i]),
        keras.layers.Dense(100, activation=parameter[i]),
        keras.layers.Dense(10, activation='softmax')
    ])

    models_mlp["model_mlp_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                    optimizer='sgd',
                    metrics=['accuracy'])

    histories_mlp["history_mlp_" + str(i)] = models_mlp["model_mlp_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))

    models_cnn["model_cnn_" + str(i)] = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation=parameter[i], padding='same', 
                        input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation=parameter[i], padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation=parameter[i], padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=parameter[i]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation=parameter[i]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
    ])

    models_cnn["model_cnn_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    histories_cnn["history_cnn_" + str(i)] = models_cnn["model_cnn_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))


# In[ ]:


epoch = 20
fig_1, ax_1 = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
epochs = np.arange(1,epoch+1,1)
ax_1[0].set_xlabel('epochs', fontsize=20)
ax_1[0].set_ylabel('validation accuracy', fontsize=20)
ax_1[1].set_xlabel('epochs', fontsize=20)
ax_1[1].set_ylabel('validation accuracy', fontsize=20)
ax_1[0].tick_params(labelsize=15)
ax_1[1].tick_params(labelsize=15)
for i in range(len(parameter)):
    index = i
    val_accuracy_cnn = histories_cnn["history_cnn_"+str(index)].history['val_accuracy']
    val_accuracy_mlp = histories_mlp["history_mlp_"+str(index)].history['val_accuracy']
    ax_1[0].plot(epochs, val_accuracy_cnn, label=parameter[i], linewidth=4)
    ax_1[1].plot(epochs, val_accuracy_mlp, label=parameter[i], linewidth=4)
    ax_1[0].set_title("Activation - CNN")
    ax_1[1].set_title("Activation - MLP")
ax_1[0].legend(fontsize=20)
ax_1[1].legend(fontsize=20)
ax_1[0].grid()
ax_1[1].grid()  
#plt.savefig('hw2_task1')


# ### Optimizers

# In[ ]:


parameter =['SGD','RMSprop','Adam']
models_mlp ={}
models_cnn ={}
histories_mlp ={}
histories_cnn ={}


for i in range(len(parameter)):
    models_mlp["model_mlp_" + str(i)] = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    models_mlp["model_mlp_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                    optimizer=parameter[i],
                    metrics=['accuracy'])

    histories_mlp["history_mlp_" + str(i)] = models_mlp["model_mlp_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))

    models_cnn["model_cnn_" + str(i)] = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='same', 
                        input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
    ])

    models_cnn["model_cnn_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                  optimizer=parameter[i],
                  metrics=['accuracy'])
    
    histories_cnn["history_cnn_" + str(i)] = models_cnn["model_cnn_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))


# In[ ]:


epoch = 20
fig_2, ax_2 = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
epochs = np.arange(1,epoch+1,1)
ax_2[0].set_xlabel('epochs', fontsize=20)
ax_2[0].set_ylabel('validation accuracy', fontsize=20)
ax_2[1].set_xlabel('epochs', fontsize=20)
ax_2[1].set_ylabel('validation accuracy', fontsize=20)
ax_2[0].tick_params(labelsize=15)
ax_2[1].tick_params(labelsize=15)
for i in range(len(parameter)):
    index = i
    val_accuracy_cnn = histories_cnn["history_cnn_"+str(index)].history['val_accuracy']
    val_accuracy_mlp = histories_mlp["history_mlp_"+str(index)].history['val_accuracy']
    ax_2[0].plot(epochs, val_accuracy_cnn, label=parameter[i], linewidth=4)
    ax_2[1].plot(epochs, val_accuracy_mlp, label=parameter[i], linewidth=4)
    ax_2[0].set_title("Optimizers - CNN")
    ax_2[1].set_title("Optimizers - MLP")
ax_2[0].legend(fontsize=20)
ax_2[1].legend(fontsize=20)
ax_2[0].grid()
ax_2[1].grid()  
#plt.savefig('hw2_task1')


# ### Regularizations (L1, L2, Dropout, no Dropout).

# #### L1,L2,L1L2

# In[ ]:


parameter =[None,'l1','l2','l1_l2']
models_mlp ={}
models_cnn ={}
histories_mlp ={}
histories_cnn ={}


for i in range(len(parameter)):
    models_mlp["model_mlp_" + str(i)] = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(300, activation='relu',kernel_regularizer =parameter[i],
        bias_regularizer=parameter[i],
        activity_regularizer=parameter[i]),
        keras.layers.Dense(100, activation='relu',kernel_regularizer =parameter[i],
        bias_regularizer=parameter[i],
        activity_regularizer=parameter[i]),
        keras.layers.Dense(10, activation='softmax',kernel_regularizer =parameter[i],
        bias_regularizer=parameter[i],
        activity_regularizer=parameter[i])
    ])

    models_mlp["model_mlp_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                    optimizer='sgd',
                    metrics=['accuracy'])

    histories_mlp["history_mlp_" + str(i)] = models_mlp["model_mlp_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))

    models_cnn["model_cnn_" + str(i)] = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='same',kernel_regularizer =parameter[i],
        bias_regularizer=parameter[i],
        activity_regularizer=parameter[i], 
                        input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same',kernel_regularizer =parameter[i],
    ),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer =parameter[i],
    ),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu',kernel_regularizer =parameter[i],
    ),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu',kernel_regularizer =parameter[i],
    ),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax',kernel_regularizer =parameter[i],
    )
    ])

    models_cnn["model_cnn_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    histories_cnn["history_cnn_" + str(i)] = models_cnn["model_cnn_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))


# In[ ]:


epoch = 20
fig_3, ax_3 = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
epochs = np.arange(1,epoch+1,1)
ax_3[0].set_xlabel('epochs', fontsize=20)
ax_3[0].set_ylabel('validation accuracy', fontsize=20)
ax_3[1].set_xlabel('epochs', fontsize=20)
ax_3[1].set_ylabel('validation accuracy', fontsize=20)
ax_3[0].tick_params(labelsize=15)
ax_3[1].tick_params(labelsize=15)
for i in range(len(parameter)):
    index = i
    val_accuracy_cnn = histories_cnn["history_cnn_"+str(index)].history['val_accuracy']
    val_accuracy_mlp = histories_mlp["history_mlp_"+str(index)].history['val_accuracy']
    ax_3[0].plot(epochs, val_accuracy_cnn, label=parameter[i], linewidth=4)
    ax_3[1].plot(epochs, val_accuracy_mlp, label=parameter[i], linewidth=4)
    ax_3[0].set_title("Regulizers - CNN")
    ax_3[1].set_title("Regulizers - MLP")
ax_3[0].legend(fontsize=20)
ax_3[1].legend(fontsize=20)
ax_3[0].grid()
ax_3[1].grid()  
#plt.savefig('hw2_task1')


# Droupout

# In[ ]:


parameter =[0,0.25,0.5,0.75]
models_mlp ={}
models_cnn ={}
histories_mlp ={}
histories_cnn ={}


for i in range(len(parameter)):
    models_mlp["model_mlp_" + str(i)] = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dropout(parameter[i]),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(parameter[i]),
        keras.layers.Dense(10, activation='softmax')
    ])

    models_mlp["model_mlp_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                    optimizer='Adam',
                    metrics=['accuracy'])

    histories_mlp["history_mlp_" + str(i)] = models_mlp["model_mlp_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))

    models_cnn["model_cnn_" + str(i)] = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='same', 
                        input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(parameter[i]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(parameter[i]),
    keras.layers.Dense(10, activation='softmax')
    ])

    models_cnn["model_cnn_" + str(i)].compile(loss='sparse_categorical_crossentropy', 
                  optimizer='Adam',
                  metrics=['accuracy'])
    
    histories_cnn["history_cnn_" + str(i)] = models_cnn["model_cnn_" + str(i)].fit(X_train, y_train, epochs=20, 
                        validation_data=(X_valid, y_valid))


# In[ ]:


epoch = 20
fig_4, ax_4 = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
epochs = np.arange(1,epoch+1,1)
ax_4[0].set_xlabel('epochs', fontsize=20)
ax_4[0].set_ylabel('validation accuracy', fontsize=20)
ax_4[1].set_xlabel('epochs', fontsize=20)
ax_4[1].set_ylabel('validation accuracy', fontsize=20)
ax_4[0].tick_params(labelsize=15)
ax_4[1].tick_params(labelsize=15)
for i in range(len(parameter)):
    index = i
    val_accuracy_cnn = histories_cnn["history_cnn_"+str(index)].history['val_accuracy']
    val_accuracy_mlp = histories_mlp["history_mlp_"+str(index)].history['val_accuracy']
    ax_4[0].plot(epochs, val_accuracy_cnn, label=parameter[i], linewidth=4)
    ax_4[1].plot(epochs, val_accuracy_mlp, label=parameter[i], linewidth=4)
    ax_4[0].set_title("Dropout - CNN")
    ax_4[1].set_title("Dropout - MLP")
ax_4[0].legend(fontsize=20)
ax_4[1].legend(fontsize=20)
ax_4[0].grid()
ax_4[1].grid()  
#plt.savefig('hw2_task1')


# ##### Final Figure

# In[ ]:


import matplotlib as mpl

backend = mpl.get_backend()
mpl.use('agg')

c1 = fig_0.canvas
c2 = fig_1.canvas
c3 = fig_2.canvas
c4 = fig_3.canvas
c5 = fig_4.canvas

c1.draw()
c2.draw()
c3.draw()
c4.draw()
c5.draw()

a1 = np.array(c1.buffer_rgba())
a2 = np.array(c2.buffer_rgba())
a3 = np.array(c3.buffer_rgba())
a4 = np.array(c4.buffer_rgba())
a5 = np.array(c5.buffer_rgba())
a = np.vstack((a1,a2,a3,a4,a5))

mpl.use(backend)
fig, ax = plt.subplots(figsize=(16, 8*5))
fig.subplots_adjust(0,0,1,1)
ax.set_axis_off()
ax.matshow(a)

#plt.savefig('hw2_task1')


# ## Hyper Parameter Search
# 

# In[ ]:


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28, 28], opt = "SGD"): 
    model = keras.models.Sequential() 
    model.add(keras.layers.Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    optimizer = eval("".join(["keras.optimizers.",opt, "(lr=learning_rate)"]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics = ['accuracy'])
    return model

keras_reg = keras.wrappers.scikit_learn.KerasClassifier(build_model)


# In[ ]:


# initial test of build model function on default inputs (chosen by common sense / examples in the book)
keras_reg.fit(X_train, y_train, epochs=100,
            validation_data=(X_valid, y_valid),
            callbacks=[keras.callbacks.EarlyStopping(patience=5, min_delta=0.001)])
keras_reg.evaluate(X_test, y_test)


# In[ ]:


# define hyper parameter search space
param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2),
        "opt": ["SGD","Adam"],
}


# In[ ]:


# random search within HP space with scikit-learn
# runs quite long: can't stop and continue inbetween interations iterations
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=300,
                batch_size=50,
                validation_data=(X_valid, y_valid),
                callbacks=[keras.callbacks.EarlyStopping(patience=3, min_delta=0.001)])


# In[ ]:


print(rnd_search_cv.best_params_) #best hyperparameters 
print(rnd_search_cv.best_score_) #best CV score/accuracy
# extract and evaluate and save best model from random search
model = rnd_search_cv.best_estimator_.model
model.evaluate(X_test, y_test)
model.save("models/MLP_brute.h5")


# Optimizing hyper paramter search with KerasTuner and Baysian Optimization:

# ### MLP

# In[ ]:


# define function to build model 
# includes the diffrent HP spaces for each parameter
# indicated by "hp." object
hp = keras_tuner.HyperParameters()
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 6)):
        if i==0:
            model.add(
                layers.Dense(
                    # choose number of neurons
                    units=hp.Int(f"units_{i}",default=100, min_value=30, max_value=515, step=25),
                    # choose activation
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))
        if i==1:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}",default=300, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))
        if i==2:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}",default=300, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))

        if i==3:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}",default=300, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))

        if i==4:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}",default=300, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))
        
        else:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}",default=100, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))
        
        # choose to include dropout layer
        if hp.Boolean(f"dropout_{i+1}", default=True):
            model.add(layers.Dropout(rate=hp.Choice(f'rate_{i+1}', default=0.25, values = [0.25,0.5,0.75])))



    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log") #choose learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

build_model(keras_tuner.HyperParameters())


# choosing tuner: https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f

# In[ ]:


# set up tuner with model build function and choosen optimization method

Tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=30,
    num_initial_points=2,
    seed=90,
    directory="Task1",
    project_name="MLP",
    #overwrite=True,
)


# In[ ]:


# shows all the individual search spaces for each hyper parameter
Tuner.search_space_summary()


# In[ ]:


# this findes the best model within the previously defind HP space
Tuner.search(X_train, y_train, epochs=200,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25)],
            validation_data=(X_valid, y_valid),batch_size=250)


# In[ ]:



# Get the top 3 models
models_mlp = Tuner.get_best_models(num_models=3)


# Build best model for Fashion MNIST 
best_model_mlp = models_mlp[0]
best_model_mlp.build(input_shape=(None, 28, 28))



# Structure of model and HP summaries 
best_model_mlp.summary()
Tuner.results_summary()
best_model_mlp.evaluate(X_test, y_test) #evaluates best model

# extracts hyper parameters (no weights) and creates untrained models (top3)
best_hp_mlp = Tuner.get_best_hyperparameters(3)
mlp_set1 = Tuner.hypermodel.build(best_hp_mlp[0])
mlp_set2 = Tuner.hypermodel.build(best_hp_mlp[1])
mlp_set3 = Tuner.hypermodel.build(best_hp_mlp[2])

#best_model_mlp.save("models/best_mlp.h5")
#best_model_mlp2.save("models/best_mlp2.h5")
#best_model_mlp3.save("models/best_mlp3.h5")



# In[ ]:


# training the untrainded model with the best set of HP to visualisze the training process 
history = mlp_set1.fit(X_train, y_train, epochs=200,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)],
            batch_size=250,validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
plt.gca().set_xlim(0,29)
plt.show()


# ### CNN

# In[ ]:


# define function to build model 
# includes the diffrent HP spaces for each parameter
# indicated by "hp." object
hp = keras_tuner.HyperParameters()
def build_model_cnn(hp):
    model = keras.Sequential()
    model.add(layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_fixed_filter',default=96, min_value=32, max_value=128, step=10),
        # adding kernel size
        kernel_size=hp.Choice('conv_fixed_kernel', values = [3,5,7]),
        #activation function
        activation='relu',
        input_shape=[28,28,1],
        padding='same',)
    )
    model.add(
    layers.MaxPooling2D(2),
    ),

    # Tune the number of layers.
    for i in range(hp.Int("num_cnn_layers",default=2, min_value=1, max_value=3)):
        if i == 0:
            model.add(
                layers.Conv2D(
                    #adding filter 
                    filters=hp.Int(f'conv_{i+1}_filter',default=256, min_value=32, max_value=512, step=16),
                    # adding kernel size
                    kernel_size=hp.Choice(f'conv_{i+1}_kernel', values = [3,5]),
                    #activation function
                    activation='relu',
                    padding='same'),
            )

        elif i == 1:
            model.add(
                layers.Conv2D(
                    #adding filter 
                    filters=hp.Int(f'conv_{i+1}_filter',default=348, min_value=32, max_value=512, step=16),
                    # adding kernel size
                    kernel_size=hp.Choice(f'conv_{i+1}_kernel', values = [3,5]),
                    #activation function
                    activation='relu',
                    padding='same'),
            )

        elif i == 2:
            model.add(
                layers.Conv2D(
                    #adding filter 
                    filters=hp.Int(f'conv_{i+1}_filter',default=348, min_value=32, max_value=512, step=16),
                    # adding kernel size
                    kernel_size=hp.Choice(f'conv_{i+1}_kernel', values = [3,5]),
                    #activation function
                    activation='relu',
                    padding='same'),
            )

        else:
            model.add(
                layers.Conv2D(
                    #adding filter 
                    filters=hp.Int(f'conv_{i+1}_filter',default=256, min_value=32, max_value=512, step=16),
                    # adding kernel size
                    kernel_size=hp.Choice(f'conv_{i+1}_kernel', values = [3,5]),
                    #activation function
                    activation='relu',
                    padding='same'),
            )

        model.add(
            layers.MaxPooling2D(
                pool_size = 2
            ),
            )
                
    model.add(layers.Flatten())        

    #choose number of dense layers 
    for i in range(hp.Int("num_dense_layers", default=2, min_value=1, max_value=3)):
        if i == 0:
            model.add(
                layers.Dense(
                    #choose number of neurons
                    units=hp.Int(f"units_{i+1}",default=128, min_value=60, max_value=515, step=20),
                    #choose activation
                    activation=hp.Choice("activation", ["relu", "tanh"]),),
            )
        
        elif i == 1:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i+1}",default=60, min_value=60, max_value=515, step=20),
                    activation=hp.Choice("activation", ["relu", "tanh"]),),
            )

        else:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i+1}",default=60, min_value=60, max_value=515, step=20),
                    activation=hp.Choice("activation", ["relu", "tanh"]),),
            )

        # choose to include dropout layer
        if hp.Boolean(f"dropout_{i+1}", default=True):
            model.add(layers.Dropout(rate=hp.Choice(f'rate_{i+1}', default=0.25, values = [0.25,0.5,0.75])))
    
    model.add(layers.Dense(10, activation="softmax")) #output layer

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

build_model_cnn(keras_tuner.HyperParameters())


# In[ ]:


# set up tuner with model build function and choosen optimization method
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model_cnn,
    objective="val_accuracy",
    max_trials=20,
    num_initial_points=2,
    seed=90,
    directory="Task1",
    project_name="CNN",
    #overwrite=True,
    #executions_per_trial,
)


# In[ ]:


# indicates search spaces for all the HPs
tuner.search_space_summary()


# In[ ]:


# this findes the best model within the previously defind HP space
tuner.search(X_train, y_train, epochs=250,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)],
            validation_data=(X_valid, y_valid), batch_size=300)


# In[ ]:


# Get the top 3 models.
models_cnn = tuner.get_best_models(num_models=3)
 

# Build best model for Fashion MNIST 
best_model_cnn = models_cnn[0]
best_model_cnn.build(input_shape=(None, 28, 28))



# summary of model structure and choosen HPS
best_model_cnn.summary()
tuner.results_summary()
best_model_cnn.evaluate(X_test, y_test)

# extracts hyper parameters (no weights) and creates untrained models (top3)
best_hp_cnn = tuner.get_best_hyperparameters(3)
cnn_set1 = tuner.hypermodel.build(best_hp_cnn[0])
cnn_set2 = tuner.hypermodel.build(best_hp_cnn[1])
cnn_set3 = tuner.hypermodel.build(best_hp_cnn[2])


# In[ ]:


# training the untrainded model with the best set of HP to visualisze the training process 
history = cnn_set1.fit(X_train, y_train, epochs=200,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)],
            batch_size=250,validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
plt.gca().set_xlim(0,29)
plt.show()


# ## CIFAR10

# In[ ]:


# load cifar 10 data
cifar10 = keras.datasets.cifar10
(x_train_full_cif, y_train_full_cif), (x_test_full_cif, y_test_full_cif) = cifar10.load_data()
X_valid_cif, X_train_cif = x_train_full_cif[:5000] / 255.0, x_train_full_cif[5000:] / 255.0
y_valid_cif, y_train_cif = y_train_full_cif[:5000], y_train_full_cif[5000:]


# In[ ]:


# fit untrained model with best set of parameter on CIFAR data
history = mlp_set1.fit(X_train_cif, y_train_cif, epochs=250,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)], 
                        validation_data=(X_valid_cif, y_valid_cif),batch_size = 200)
pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()
mlp_set1.evaluate(x_test_full_cif, y_test_full_cif)


# In[ ]:


# fit untrained model with second best set of parameter on CIFAR data
history = mlp_set2.fit(X_train_cif, y_train_cif,epochs=250,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)], 
                        validation_data=(X_valid_cif, y_valid_cif),batch_size = 200)
pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()
mlp_set2.evaluate(x_test_full_cif, y_test_full_cif)


# In[ ]:


# fit untrained model with third best set of parameter on CIFAR data
history = mlp_set3.fit(X_train_cif, y_train_cif, epochs=250,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)], 
                        validation_data=(X_valid_cif, y_valid_cif),batch_size = 200)
pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()
mlp_set3.evaluate(x_test_full_cif, y_test_full_cif)


# ### CNN

# In[ ]:


# get structure and hyper parameters of best CNN model on MNIST data 
cnn_set1.summary()
tuner.results_summary(1)


# In[ ]:


# rebuild model with best set of parameters changing it to allow for diffrently shaped data
cifar_cnn1 = keras.models.Sequential([
    keras.layers.Conv2D(102, 3, activation='relu', padding='same', 
                        input_shape=[32,32,3]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(160, 5, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(160, activation='tanh'),
    keras.layers.Dense(10, activation='softmax')
])

cifar_cnn1.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=keras.optimizers.Adam(learning_rate=0.00014948684422968163),
                  metrics=['accuracy'])

history = cifar_cnn1.fit(X_train_cif, y_train_cif, epochs=250,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)], 
                        validation_data=(X_valid_cif, y_valid_cif),batch_size = 200)

pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()
cifar_cnn1.evaluate(x_test_full_cif, y_test_full_cif)


# In[ ]:


# get structure and hyper parameters of second best CNN model on MNIST data 
cnn_set2.summary()
tuner.results_summary(2)


# In[ ]:


# rebuild model with second best set of parameters changing it to allow for diffrently shaped data
cifar_cnn2 = keras.models.Sequential([
    keras.layers.Conv2D(92, 3, activation='relu', padding='same', 
                        input_shape=[32,32,3]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(512, 5, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(80, activation='tanh'),
    keras.layers.Dense(10, activation='softmax')
])

cifar_cnn2.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001
),
                  metrics=['accuracy'])

history = cifar_cnn2.fit(X_train_cif, y_train_cif, epochs=250,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)], 
                        validation_data=(X_valid_cif, y_valid_cif),batch_size = 200)

pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()
cifar_cnn2.evaluate(x_test_full_cif, y_test_full_cif)


# In[ ]:


# get structure and hyper parameters of third best CNN model on MNIST data 
cnn_set3.summary()
tuner.results_summary(3)


# In[ ]:


# rebuild model with third best set of parameters changing it to allow for diffrently shaped data
cifar_cnn3 = keras.models.Sequential([
    keras.layers.Conv2D(92, 3, activation='relu', padding='same', 
                        input_shape=[32,32,3]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(224, 5, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(400, activation='tanh'),
    keras.layers.Dense(10, activation='softmax')
])

cifar_cnn3.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001
),
                  metrics=['accuracy'])

history = cifar_cnn3.fit(X_train_cif, y_train_cif, epochs=250,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)], 
                        validation_data=(X_valid_cif, y_valid_cif),batch_size = 200)

pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()
cifar_cnn3.evaluate(x_test_full_cif, y_test_full_cif)

