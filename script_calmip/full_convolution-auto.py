#!/usr/bin/env python
# coding: utf-8

# ## Entrainement de plusieurs models et plusieurs dataset au choix
# 
# 
# 
###il renvoie un rapport en json
###le datagen reste toujours "non" avant que je fais un fonction pour que le data d'image est en 4 dimension pour adapter le fonction de dataugmentation de keras
# ## Import Python Module

# In[ ]:


#gestion fichier modeles pythons
import os , sys
from os import path

#modules utilitaires
import random as r
import numpy as np
import time 
import json
import pwk   
#traitement image
import matplotlib.pyplot as plt   
import skimage as sm
from skimage import io 
import pandas as pd
import glob
from pathlib import Path
from IPython.display import display, Markdown
import bdd_prs as bdd

#module IA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
from tensorflow.keras import activations


# In[ ]:


#vérification d'existance de path, pas nécessaire à executer
workpath = sys.path[0]
mon_repertoire = f'{workpath}/..'
path_exit = path.isdir(mon_repertoire)
print(f"path mon_repertoire existe? {path_exit}")


# In[ ]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# ## les diffs choix de dataset.

# In[ ]:


datapath = f"{mon_repertoire}/data_128_equilibre"
descri_datapath = [datapath, 'img_128']


# ## les path à stocker les models entrainées, leur fichiers de tensorborad et de json

# In[ ]:


run_dir=f"{mon_repertoire}/trained_model_img_models/model_128_equilibre"
os.makedirs(run_dir, mode=0o700, exist_ok=True)


# ## Parameters for CNN and DNN

# In[ ]:


#pourcentage du dataset à load
scale = 1

batch_size    = 64
epochs        = 15
fit_verbosity = 1

index_to_class = ["CUBO","DEC","FCC","FCC-sphere","HCP-sphere","ICO","MnBeta_sphere","OH","RTD","BCC","DODECA"]


# ## Dataset loading

# In[ ]:


def read_dataset (filepath : str):
    """entrée : chemin vers le dossier contenant la BDD
    sortie : listes des images et leur classes respectives, mis à l'échelle (scale) et mélangées"""
    
    """
    au lieu d'avoir un set de training 80%, un set de test 20%, j'ai appliqué ici un strategie conseillé si on a beacoup de donnee, 
    un training set 70%, un test set pour validation 15%, et un test final set 15% pour faire l'evaluation. Vu que le test set a une effet parasit sur le training set
    """
    
    L = os.listdir(filepath)   #liste contenenant le nom (en .jpg) de toutes les images
    nb_files = len(L)
    nb_files2load = round(nb_files*scale)
    size = np.shape(io.imread(f'{filepath}/{L[0]}' , as_gray = True))
    
    #initialisation listes sortie
    img = np.zeros ( (nb_files2load , size[0] , size[1]) , dtype = np.float16 )
    ID  = np.zeros ( nb_files2load , dtype = np.int8)
   
    for i in range (nb_files2load):
        
        random_index = np.random.randint(0 , nb_files - i )
        img[i] = io.imread (f'{filepath}/{L[random_index]}',as_gray = True)
        ID [i] = int (L[random_index][:3]) - 1
        L.pop(random_index)    
        
    N = len(img)
    nb_img = N
    a = N*0.7
    b = N*0.85
    img_train = img[0:round(a)]
    img_test  = img[round(a):round(b)]
    img_test_final = img[round(b):]
    ID_train = ID[0:round(a)]
    ID_test  = ID[round(a):round(b)]  
    ID_test_final = ID[round(b):]
    
        
    return img_train,img_test,ID_train,ID_test,nb_img, img_test_final, ID_test_final


# ## dataugmentation

# In[ ]:


def dataugmentation(ID : dict , img_train , ID_train): 
    """
    ID est les classes et les taux d'augmentation respective 
    """
    for Id in ID: # pour tester les entrees sont bonne
        assert 0 <= Id <= 10, ("le classe doit etre compris entre 0 et 10")
        assert 0 <= ID[Id] <= 1, ("le taux d'augmentation doit etre compris entre 0 et 1")  
    L = len(img_train)
    L1 = len(ID_train)
    assert L == L1, ("array image et son ID doit avoir la meme longeur")
    #balayer tout les images
    for i in range(L-1):
        y = ID_train[i]
        img = img_train[i]
        #cette image correspond-t-il a un classe on veux dataugmenter? 
        for Id in ID:
            if y == Id:
                img_ro = bdd.light_modif(img)
                
                #insertion d'image creer et son ID dans un position random 
                random_index = np.random.randint(0,L)
                img_train = np.insert(img_train, random_index, img_ro, axis = 0)
                ID_train = np.insert(ID_train, random_index, Id, axis = 0) 

    print(f'longeur actuelle est {img_train.shape[0]}, longeur initial est {L}')            
    return img_train, ID_train


                
                
    
    


# ## Models creations

# ## pour tester la reprodubilite de resultat de cnn (tu peux mettre qq meme structure de cnn mais les nomme differament)

# ## les models ou on convolue un truc plusieur fois devant un maxpooling

# In[ ]:


def m2_700_112234(lx,ly): #32:1, 64:2, 128:3, 256:4, ailleur, le padding me permet d'avoir le couche 256
    model = keras.models.Sequential()
    
    model.add( keras.layers.Conv2D(32, (3,3), padding = 'same',  activation='relu', input_shape=(lx,ly,1)))
    model.add( keras.layers.Conv2D(32, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((3, 3)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(128, (3, 3),padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))
    
    model.add( keras.layers.Conv2D(256, (3, 3),padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Flatten()) 
    model.add( layers.Dense(700))
    model.add( layers.Dense(700))
    model.add( layers.Activation(activations.tanh)) 
    model.add( keras.layers.Dropout(0.3))

    model.add( keras.layers.Dense(11, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    code_model='2_700_112234'
    model_name = 'e'+str(epochs)+'bs'+str(batch_size)+'sc'+str(scale)+'code'+str(code_model)
    return model, model_name, code_model

def m2_700_11234(lx,ly): #32:1, 64:2, 128:3, 256:4
    model = keras.models.Sequential()
    
    model.add( keras.layers.Conv2D(32, (3,3), padding = 'same',  activation='relu', input_shape=(lx,ly,1)))
    model.add( keras.layers.Conv2D(32, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((3, 3)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(128, (3, 3),padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))
    
    model.add( keras.layers.Conv2D(256, (3, 3),padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Flatten()) 
    model.add( layers.Dense(700))
    model.add( layers.Dense(700))
    model.add( layers.Activation(activations.tanh)) 
    model.add( keras.layers.Dropout(0.3))

    model.add( keras.layers.Dense(11, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    code_model='2_700_11234'
    model_name = 'e'+str(epochs)+'bs'+str(batch_size)+'sc'+str(scale)+'code'+str(code_model)
    return model, model_name, code_model


def m_ince_11234(lx,ly): # _ince: en applicant les kernel 1*1
    model = keras.models.Sequential()
    
    model.add( keras.layers.Conv2D(16, (1,1), padding = 'same',  activation='relu', input_shape=(lx,ly,1)))
    model.add( keras.layers.Conv2D(32, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.Conv2D(32, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((3, 3)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(128, (3, 3),padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))
    
    model.add( keras.layers.Conv2D(256, (3, 3),padding = 'same', activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Flatten()) 
    model.add( layers.Dense(700))
    model.add( layers.Dense(700))
    model.add( layers.Activation(activations.tanh)) 
    model.add( keras.layers.Dropout(0.3))

    model.add( keras.layers.Dense(11, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    code_model='2_700_11234'
    model_name = 'e'+str(epochs)+'bs'+str(batch_size)+'sc'+str(scale)+'code'+str(code_model)
    return model, model_name, code_model


# ## qq variable a entrer dans les foncitons suivants (n'oublier pas de changer tag_id si on change de modele, dataset, etc.)

# In[ ]:


# les entree pour le dataset 64
datasets = [descri_datapath]
with_datagen  = [False] #pour diviser l'entrainement en 2 cas: avec et sans dataugmentation, si tu ne veux qu'un cas, supprimes l'element que tu veux pas
t = pwk.tag_now()
tag_id = f'cnn_fVGG_{str(t)}' #si on faire un autre execution de script, n'oublie pas de changer le tag_id comme "fonc_acti_<un date>" ou "nombre_de_couche", etc.
#tag_id = 'optimizer_08_03_v2'
#en fait, on peut lance plusieurs fois les meme entrainements, pour surperposer les courbes sur le meme schema dans tensorboard.
#models = ['create_model1','create_model1_DNNsigmod','create_model1_DNNtanh','create_model1_DNNsoftsign','create_model1_DNNelu','create_model1_DNNselu','create_model1_DNNsoftplus']
#models = ['create_model1_100','create_model2_100','create_model3_100']
#models = ['create_model1_DNNtanh_adam','create_model1_DNNtanh_Adamax','create_model1_DNNtanh_Nadam','create_model1_DNNtanh_RMSprop']
#models = ['create_model2_100','create_model2_300','create_model2_500','create_model2_700','create_model2_900','create_model2_1000']
#models = ['m2_2000_dr_01','m2_2000_dr_02','m2_2000_dr_03','m2_2000_dr_04','m2_2000_dr_05']
#models  = ['m2_700_pad_4','m2_700_pad_4_1','m2_700_pad_4_2','m2_700_pad_4_3']
models = ['m2_700_112234']


# In[ ]:


ID_bad = {4 : 0.2} # parametre de dataugmentation. Le premier est la ID de la classe, le 2eme est le taux de production de nouveaux image de ce ID


# ## Multiple datasets, multiple models ;-)

# In[ ]:


def multi_run(datasets, models, with_datagen = [False],
              scale=1, batch_size=64, epochs=16, 
              fit_verbosity=0, tag_id='last'):
    """
    Launches a dataset-model combination
    args:
        enhanced_dir   : Directory of the enhanced datasets
        datasets       : List of dataset (whitout .h5)
        models         : List of model like { "model name":get_model(), ...}
        datagen        : Data generator or None (None)
        scale          : % of dataset to use.  1 mean all. (1)
        batch_size     : Batch size (64)
        epochs         : Number of epochs (16)
        fit_verbosity  : Verbose level (0)
        tag_id         : postfix for report, logs and models dir (_last)
    return:
        report        : Report as a dict for Pandas.
    """  
    # ---- Logs and models dir
    #
    os.makedirs(f'{run_dir}/logs_{tag_id}',   mode=0o750, exist_ok=True)
    os.makedirs(f'{run_dir}/models_{tag_id}', mode=0o750, exist_ok=True)
    
    # ---- Columns of output
    #
    output={}
    output['Dataset'] = []
    output['Size']    = []
    
    for m in models:
        output[m+'_sdg_Accuracy'] = []
        output[m+'_sdg_Duration'] = []
        output[m+'_sdg_loss'] = []
        output[m+'_wdg_Accuracy'] = []
        output[m+'_wdg_Duration'] = []
        output[m+'_wdg_loss'] = []
    # ---- Let's go
    #
    for d_name in datasets:
        print("\nDataset : ",d_name[0])
        
        # ---- Read dataset
        img_train,img_test,ID_train,ID_test, d_size, img_test_final, ID_test_final = read_dataset(d_name[0])
        imgs = [img_train,img_test,ID_train,ID_test, img_test_final, ID_test_final]
        d_name = d_name[1]
        output['Dataset'].append(d_name)
        output['Size'].append(d_size)
        print(len(img_train))
        print(len(ID_train))
        # ---- Rescale c'est pas vraiment necessaire
        ##img_train,ID_train,img_test,ID_test = pwk.rescale_dataset(img_train,ID_train,img_test,ID_test, scale=scale)
        
        # ---- Get the shape
        (n,lx,ly) = img_train.shape

        # ---- For each model
        for m_function in models:
            
            # ---- get model
            try:
                # ---- get function by name
                m = m_function
                m_function=globals()[m_function]
                model_wdg, m_name, code_model=m_function(lx,ly)
                model_sdg = m_function(lx,ly)[0]
                print("    Run model {}  : ".format(code_model), end='')
                # ---- Compile it # si on ne compare pas de diffs optimizer on ajoute ce ligne.
                if tag_id[:6] != 'opitmizer':
                    model_wdg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    model_sdg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                # ---- Callbacks tensorboard
                #_wdg == avec dataugmentation, _sdg == sans dataugmentation
                log_dir_wdg = f'{run_dir}/logs_{tag_id}/tb_wdg_{d_name}_{m_name}'
                tensorboard_callback_wdg = tf.keras.callbacks.TensorBoard(log_dir=log_dir_wdg, histogram_freq=1)
                
                log_dir_sdg = f'{run_dir}/logs_{tag_id}/tb_sdg_{d_name}_{m_name}'
                tensorboard_callback_sdg = tf.keras.callbacks.TensorBoard(log_dir=log_dir_sdg, histogram_freq=1)
                # ---- Callbacks bestmodel
                save_dir_wdg = f'{run_dir}/models_{tag_id}/model_wdg_{d_name}_{m_name}.h5'
                bestmodel_callback_wdg = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir_wdg, verbose=0, monitor='accuracy', save_best_only=True)
                
                save_dir_sdg = f'{run_dir}/models_{tag_id}/model_sdg_{d_name}_{m_name}.h5'
                bestmodel_callback_sdg = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir_sdg, verbose=0, monitor='accuracy', save_best_only=True)
                # ---- Train
                
                for datagen in with_datagen:
                    start_time = time.time()
                    # ---- No data augmentation (datagen=None) --------------------------------------
                    if datagen==False:
                        print("sans dataugmentation")
                        history = model_sdg.fit(img_train, ID_train,
                                            batch_size      = batch_size,
                                            epochs          = epochs,
                                            verbose         = fit_verbosity,
                                            validation_data = (img_test, ID_test),
                                            callbacks       = [tensorboard_callback_sdg, bestmodel_callback_sdg])
                        end_time = time.time()
                        duration = end_time-start_time
                        loss, accuracy = model_sdg.evaluate(img_test_final, ID_test_final, batch_size=batch_size)
                        #resultat pour fichier json
                        output[m+'_sdg_Accuracy'].append(accuracy)
                        output[m+'_sdg_Duration'].append(duration)
                        output[m+'_sdg_loss'].append(loss)
                        print(f"Accuracy={accuracy: 7.2f}    Duration={duration: 7.2f}")
                        
                    else:
                        print("with dataugmentation")
                         # ---- Data augmentation (datagen given) ----------------------------------------
                            
                        img_train_wdg, ID_train_wdg = dataugmentation(ID = ID_bad , img_train = img_train , ID_train = ID_train)
                        history = model_wdg.fit(img_train_wdg, ID_train_wdg,
                                            batch_size      = batch_size,
                                            epochs          = epochs,
                                            verbose         = fit_verbosity,
                                            validation_data = (img_test, ID_test),
                                            callbacks       = [tensorboard_callback_wdg, bestmodel_callback_wdg])
                        end_time = time.time()
                        duration = end_time-start_time
                        loss, accuracy = model_wdg.evaluate(img_test_final, ID_test_final, batch_size=batch_size)
                        #resultats pour fichier json 
                        output[m+'_wdg_Accuracy'].append(accuracy)
                        output[m+'_wdg_Duration'].append(duration)
                        output[m+'_wdg_loss'].append(loss)
                        print(f"Accuracy={accuracy: 7.2f}    Duration={duration: 7.2f}")
                verification_array = np.array([m+'_sdg_Accuracy',m+'_sdg_Duration',m+'_sdg_loss',m+'_wdg_Accuracy',m+'_wdg_Duration',m+'_wdg_loss'])      
                for casse in verification_array:
                    if output[casse] == []:
                        output[casse].append('-')
            except:
                raise
                print('-')
                print("comment c'est fucking possible que ce l'entrainement ne marche pas")
            
    return output, imgs


# ## run

# In[ ]:


pwk.chrono_start()

print('\n---- Run','-'*50)


# ---- Data augmentation or not

    
# ---- Run
#
output, imgs = multi_run(datasets, 
                   models,
                   with_datagen  = with_datagen,
                   scale         = scale,
                   batch_size    = batch_size,
                   epochs        = epochs,
                   fit_verbosity = fit_verbosity,
                   tag_id        = tag_id)

# ---- Save report
#
report={}
report['output']=output
report['description'] = f'scale={scale} batch_size={batch_size} epochs={epochs} data_aug={with_datagen}'

report_name=f'{run_dir}/report_{tag_id}.json'

with open(report_name, 'w') as file:
    json.dump(report, file, indent=4)

print('\nReport saved as ',report_name)

pwk.chrono_show()
print('-'*59)


# ## show report

# In[ ]:


report_dir = run_dir


# In[ ]:


def highlight_max(s):
    is_max = (s == s.max())
    return ['background-color: yellow' if v else '' for v in is_max]

def show_report(file):
    # ---- Read json file
    with open(file) as infile:
        dict_report = json.load( infile )
    output      = dict_report['output']
    description = dict_report['description']
    # ---- about
    pwk.subtitle(f'Report : {Path(file).stem}')
    print(    "Desc.  : ",description,'\n')
    # ---- Create a pandas
    report       = pd.DataFrame (output)
    col_accuracy = [ c for c in output.keys() if c.endswith('Accuracy')]
    col_duration = [ c for c in output.keys() if c.endswith('Duration')]
    # ---- Build formats
    lambda_acc = lambda x : '{:.2f} %'.format(x) if (isinstance(x, float)) else '{:}'.format(x)
    lambda_dur = lambda x : '{:.1f} s'.format(x) if (isinstance(x, float)) else '{:}'.format(x)
    formats = {'Size':'{:.2f} Mo'}
    for c in col_accuracy:   
        formats[c]=lambda_acc
    for c in col_duration:
        formats[c]=lambda_dur
    t=report.style.highlight_max(subset=col_accuracy).format(formats).hide_index()
    display(t)


# ## Step 3 - Reports display

# In[ ]:


#for file in glob.glob(f'{report_dir}/*.json'):
file = report_name
show_report(file)


# ## model evaluate matrix de confusion 

# In[ ]:


models = os.listdir(f'{run_dir}\models_{tag_id}')
print(models) # tu choisis un des modeles listés et le mettre dans le variable model_choisit au dessous.


# In[ ]:


model_choisit = 'model_wdg_img_64_e15bs64sc0.5code2_700_11234.h5'
model_choisit = f'{run_dir}\models_{tag_id}\{model_choisit}'
model_evaluated = tf.keras.models.load_model(model_choisit) #load ton model choisite, tu peux aussi copie coller la chemin absolute ou le model est stocké


# In[ ]:


model_evaluated.summary() # son structure


# In[ ]:


img_test_final = imgs[4]
ID_test_final = imgs[5]
score = model_evaluated.evaluate(img_test_final, ID_test_final, verbose=0) #evaluation de ce models

print('Test loss     :', score[0])
print('Test accuracy :', score[1])


# ## erreurs

# In[ ]:


y_sigmoid = model_evaluated.predict(img_test_final)
ID_pred    = np.argmax(y_sigmoid, axis=-1)

pwk.plot_images(img_test_final, ID_test_final, range(0,200), columns=12, x_size=1, y_size=1, y_pred=ID_pred, save_as='04-predictions')


# In[ ]:


pwk.plot_confusion_matrix(ID_test_final,ID_pred,range(11),normalize=True, save_as='06-confusion-matrix')


# In[ ]:




