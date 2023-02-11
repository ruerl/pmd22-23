# %% [markdown]
# ## Import

# %%
import os,sys

import random as r
import numpy as np

import matplotlib.pyplot as plt
from skimage import io  , util
from skimage import transform as t
from skimage.filters import gaussian

# %% [markdown]
# ## Modif Image

# %%
def move_matrix (matrix  , x_up = 0 , x_down = 0 , x_left = 0, x_right = 0):
    """Renvoi l'image déplacée suivant les 4 directions """
    
    px , py = np.shape(matrix)
    h = int (abs ( x_up - x_down ))
    v = int (abs ( x_right - x_left ))
    
    #Sous-fonctions de déplacement
    def up(matrix , x_up : int):
        deplaced_matrix = np.zeros ( (px , py) , dtype = float)
        for i in range(px - x_up):
            deplaced_matrix[i] = matrix[i + x_up]
        return deplaced_matrix
    
    def down(matrix , x_down: int):
        deplaced_matrix = np.zeros ( (px , py) , dtype = float)
        for i in range(x_down, px):
            deplaced_matrix[i] = matrix[i - x_down]
        return deplaced_matrix
    
    def left(matrix , x_left : int):
        deplaced_matrix = np.zeros ( (px , py) , dtype = float)
        for i in range(px):
            for j in range (py - x_left):
                deplaced_matrix[i][j] = matrix[i][j + x_left]
        return deplaced_matrix
    
    def right(matrix , x_right : int):
        deplaced_matrix = np.zeros ( (px , py) , dtype = float)
        for i in range(px):
            for j in range (x_right, py):
                deplaced_matrix[i][j] = matrix[i][j - x_right]
        return deplaced_matrix
    
    #Choix du cadran
    if x_up >= x_down and x_right >= x_left :
        return up  (right(matrix,v),h)
    
    if x_up <= x_down and x_right >= x_left :
        return down(right(matrix,v),h)
    
    if x_up >= x_down and x_right <= x_left :
        return up  (left (matrix,v),h)
    
    if x_up <= x_down and x_right <= x_left :
        return down(left (matrix,v),h)
    
def frame_NP(matrix):
    """Renvoi les indices des lignes et colonnes du cadre contenant la particule""" 
    
    px , py = np.shape(matrix)
    i_start = 0
    i_end   = 0
    j_start = 0
    j_end   = 0
    
    i = 0
    stop = False
    while i < px and stop == False:
        j = 0
        while j < py and stop == False:
            if matrix[i][j] > 0.1:
                i_start = i-1
                stop = True
            j += 1
        i += 1
    
    i = px - 1
    stop = False
    while i > 0 and stop == False:
        j = 0
        while j < py and stop == False:
            if matrix[i][j] > 0.1:
                i_end = i+1
                stop = True
            j += 1
        i -= 1
        
    j = 0
    stop = False
    while j < py and stop == False:
        i = 0
        while i < px and stop == False:
            if matrix[i][j] > 0.1:
                j_start = j - 1
                stop = True
            i += 1
        j += 1
    
    j = py - 1
    stop = False
    while j > 0 and stop == False:
        i = 0
        while i < px and stop == False:
            if matrix[i][j] > 0.1:
                j_end = j + 1
                stop = True
            i += 1
        j -= 1   
    
    return i_start , i_end , j_start , j_end

def draw_frame ( matrix):
    px , py = np.shape(matrix)
    i_start , i_end , j_start , j_end = frame_NP(matrix)
    f_matrix = np.copy(matrix)
    for i in range(px):
        for j in range(py):
            if (i == i_start or i == i_end) or (j == j_start or j == j_end):
                f_matrix[i][j] = 1.0
    return f_matrix

def noise (matrix):
    px , py = np.shape(matrix)
    noised_matrix = np.copy(matrix)
    for i in range (px):
        for j in range (py):
            if matrix[i][j] < 0.8:
                noised_matrix[i][j] = r.random()
    return  noised_matrix 

# %%
def light_modif (matrix):
    px , py = np.shape(matrix)
    
    m_matrix = t.rotate (matrix , r.randint(0,360) )
    
    i_start, i_end , j_start , j_end = frame_NP(m_matrix)
    x_up    = r.randint(0 , i_start)
    x_down  = r.randint(0 , px - 1 - i_end)
    x_left  = r.randint(0 , j_start)
    x_right = r.randint(0 , py - 1 - j_end)
    
    m_matrix = move_matrix(m_matrix , x_up , x_down , x_left , x_right)
    
    return m_matrix

def hard_modif (matrix , n = True):
    px , py = np.shape(matrix)
    
    m_matrix = t.rotate(matrix , r.random()*360) 
    
    i_start , i_end , j_start , j_end = frame_NP(m_matrix)
    lx = (px - 1 - i_end)
    ly = (py - 1 - j_end)
    l  = r.randint(min(lx , ly, i_start , j_start)//2 , min(lx , ly, i_start , j_start))
    
    m_matrix = util.crop( m_matrix , ((i_start - l , lx - l ) , (j_start - l , ly - l )) , copy = True)
    
    i_size = np.shape(m_matrix)
    
    x_up    = r.randint(0,i_size[0]//2)
    x_down  = r.randint(0,i_size[0]//2)
    x_left  = r.randint(0,i_size[1]//2)
    x_right = r.randint(0,i_size[1]//2)
    sigma   = (r.random() * 3) + 1
                           
    m_matrix = move_matrix(m_matrix , x_up , x_down , x_left , x_right)   
    m_matrix = t.resize ( m_matrix , (px,py))
    if n == True:
        m_matrix = noise(m_matrix)
    m_matrix = gaussian (m_matrix , sigma)
    
    return m_matrix

# %% [markdown]
# ## Modif BDD

# %%
def empty_BDD(BDDpath : str):
    for img_name in os.listdir(BDDpath):
        os.remove(f'{BDDpath}\{img_name}')

# %%
def augmente_BDD (BDDpath : str, mode : bool, img2add = 0, id2exlude = []):
    
    BDDcontent = os.listdir(BDDpath)
    Nbimg = len(BDDcontent)
    
    if mode == True:
        i = 0
        while i < img2add:
            random_index = r.randrange(0 , Nbimg)
            img_name = BDDcontent[random_index]
            if int(img_name[0:3]) not in id2exlude:
                img_matrix = io.imread (f'{BDDpath}\{img_name}' , as_gray = True)
                img_name = img_name[:-4] + f'µ{i}.jpg'
                m_img_matrix = light_modif(img_matrix)
                plt.imsave (f'{BDDpath}\{img_name}' , m_img_matrix , cmap = 'gray' , format = 'jpg')
                i+=1 
        
    if mode == False:
        for img_name in BDDcontent:
            if 'µ' in img_name:
                os.remove(f'{BDDpath}\{img_name}')

# %%
def resize_BDD (BDDpath : str , size : int):
    """Entrée : chemin de la BDD , taille de l'image carré de sortie .
    Sortie : None , Stockes les images dans un dossier crée à cet effet, dans le même dossier que le fichier source """
    
    BDD_name = str.split (BDDpath , '\\')[-1]
    root = BDDpath[:-len(BDD_name) - 1] 
    BDDcontent = os.listdir(BDDpath)
    
    if not os.path.exists(f'{root}\{BDD_name}({size})'):  
        os.makedirs(f'{root}\{BDD_name}({size})')
    
    if len(os.listdir(f'{root}\{BDD_name}({size})')) == 0:
        for img_name in BDDcontent :
            img = io.imread(f'{BDDpath}\{img_name}' , as_gray = True)
            im_red = t.resize(img, (size,size))
            plt.imsave(f'{root}\{BDD_name}({size})\{img_name}' , im_red , cmap = 'gray' , format='jpg')         

# %%
def real_BDD (BDDpath : str , n = True):
    
    BDD_name = str.split (BDDpath , '\\')[-1]
    root = BDDpath[:-len(BDD_name) - 1]
    BDDcontent = os.listdir(BDDpath)
    
    if n==True:
        suffix = '-noise'
    else:
        suffix = '-no-noise'
    
    if not os.path.exists(f'{root}\{BDD_name}(real{suffix})'):  
        os.makedirs(f'{root}\{BDD_name}(real{suffix})')
    
    if len(os.listdir(f'{root}\{BDD_name}(real{suffix})')) == 0:
        for img_name in BDDcontent :
            img = io.imread(f'{BDDpath}\{img_name}' , as_gray = True)
            m_img = hard_modif(img,n)
            plt.imsave(f'{root}\{BDD_name}(real{suffix})\{img_name}' , m_img , cmap = 'gray' , format='jpg')

# %%



