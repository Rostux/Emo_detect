#
# Basic Imports
from sklearn.model_selection import KFold
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import linalg

# Loading and plotting data
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Features
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import _class_means, _class_cov
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[49]:
opt = {
    'image_size': 32,
    'is_grayscale': False,
    'val_split': 0.75
}


# In[50]:


cfw_dict = {'Amitabhbachan': 0,
            'AamirKhan': 1,
            'DwayneJohnson': 2,
            'AishwaryaRai': 3,
            'BarackObama': 4,
            'NarendraModi': 5,
            'ManmohanSingh': 6,
            'VladimirPutin': 7}

imfdb_dict = {'MadhuriDixit': 0,
              'Kajol': 1,
              'SharukhKhan': 2,
              'ShilpaShetty': 3,
              'AmitabhBachan': 4,
              'KatrinaKaif': 5,
              'AkshayKumar': 6,
              'Amir': 7}


def load_image(path):
    im = Image.open(path).convert('L' if opt['is_grayscale'] else 'RGB')
    im = im.resize((opt['image_size'], opt['image_size']))
    im = np.array(im)
    im = im/256
    return im


def load_data(dir_path):
    image_list = []
    y_list = []

    if "CFW" in dir_path:
        label_dict = cfw_dict

    elif "yale" in dir_path.lower():
        label_dict = {}
        for i in range(15):
            label_dict[str(i+1)] = i
    elif "IMFDB" in dir_path:
        label_dict = imfdb_dict
    else:
        raise KeyError("Dataset not found.")

    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".png"):
            im = load_image(os.path.join(dir_path, filename))
            y = filename.split('_')[0]
            y = label_dict[y]
            image_list.append(im)
            y_list.append(y)
        else:
            continue

    image_list = np.array(image_list)
    y_list = np.array(y_list)

    print("Dataset shape:", image_list.shape)

    return image_list, y_list


def disply_images(imgs, classes, row=1, col=2, w=64, h=64):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, col*row + 1):
        img = imgs[i-1]
        fig.add_subplot(row, col, i)

        if opt['is_grayscale']:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        plt.title("Class:{}".format(classes[i-1]))
        plt.axis('off')
    plt.show()


# In[51]:


# eg.
dirpath = './dataset/IMFDB/'
X, y = load_data(dirpath)
N, H, W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]


# In[52]:


ind = np.random.randint(0, y.shape[0], 6)
disply_images(X[ind, ...], y[ind], row=2, col=3)
X = X.reshape((N, H*W*C))

# In[54]:


def get_pca(X, k):
    """
        Get PCA of K dimension using the top eigen vectors 
    """
    pca = PCA(n_components=k)
    X_k = pca.fit_transform(X)
    return X_k

# In[55]:


def get_kernel_pca(X, k, kernel='rbf', degree=3):
    """
        Get PCA of K dimension using the top eigen vectors 
        @param: X => Your data flattened to D dimension
        @param: k => Number of components
        @param: kernel => which kernel to use (“linear” | “poly” | “rbf” | “sigmoid” | “cosine” )
        @param: d => Degree for poly kernels. Ignored by other kernels
    """
    kpca = KernelPCA(n_components=k, kernel=kernel, degree=degree)
    X_k = kpca.fit_transform(X)
    return X_k
# In[56]:


def get_lda(X, y, k):
    """
        Get LDA of K dimension 
        @param: X => Your data flattened to D dimension
        @param: k => Number of components
    """
    lda = LDA(n_components=k)
    X_k = lda.fit_transform(X, y)
    return X_k


# In[57]:


def get_kernel_lda(X, y, k, kernel='rbf', degree=3):
    """
        Get LDA of K dimension 
        @param: X => Your data flattened to D dimension
        @param: k => Number of components
        @param: kernel => which kernel to use ( “poly” | “rbf” | “sigmoid”)
    """
    # Transform  input
    if kernel == "poly":
        X_transformed = X**degree
    elif kernel == "rbf":
        var = np.var(X)
        X_transformed = np.exp(-X/(2*var))
    elif kernel == "sigmoid":
        X_transformed = np.tanh(X)
    else:
        raise NotImplementedError("Kernel {} Not defined".format(kernel))

    klda = LDA(n_components=k)
    X_k = klda.fit_transform(X, y)
    return X_k


def get_vgg_features(dirpath):
    features = np.load(os.path.join(dirpath, "VGG19_features.npy"))
    return features


def get_resnet_features(dirpath):
    features = np.load(os.path.join(dirpath, "resnet50_features.npy"))
    return features


# In[60]:


class mlp_mod(mlp):
    def _init_coef(self, fan_in, fan_out):
        if self.activation == 'logistic':

            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation in ('identity', 'tanh', 'relu'):
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        else:
            raise ValueError("Unknown activation function %s" %
                             self.activation)

        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        coef_ini = np.random.normal(0, 1, (fan_in, fan_out))
        coef_init[0:fan_in, 0:fan_out] = coef_ini[0:fan_in, 0:fan_out]
        coef_init = coef_init / np.sqrt(fan_in+fan_out)
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        intercept_ini = np.random.normal(-init_bound,
                                         init_bound,  fan_out)
        intercept_init[0:fan_out] = intercept_ini[0:fan_out]
        intercept_init = intercept_init / np.sqrt(fan_in+fan_out)
        return coef_init, intercept_init


classifier = [mlp_mod(solver='adam', max_iter=1500, learning_rate='adaptive', hidden_layer_sizes=(100, 100), activation='relu'), SVC(kernel='rbf', gamma='auto', C=1),
              LogisticRegression(solver='lbfgs', multi_class='multinomial')]


# In[61]:


Dst1 = './dataset/IIIT-CFW'
Dst2 = './dataset/Yale_face_database'
Dst3 = './dataset/IMFDB'


# In[63]:


aaa = get_resnet_features(Dst1)

x1, y1 = load_data(Dst1)
X_train, X_test, y_train, y_test = train_test_split(aaa, y1, test_size=0.2)
for clf in classifier:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    score2 = clf.score(X_train, y_train)
    rec_score = recall_score(y_test, y_pred, average='weighted')
    acc_score = accuracy_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred, average='weighted')
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Accuracy:", acc_score)


# In[ ]:


KNN = KNeighborsClassifier(n_neighbors=5)

# In[65]:


X_train, X_test, y_train, y_test = train_test_split(aaa, y1, test_size=0.2)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
print(accuracy_score(y_test, y_pred))
acc_score = accuracy_score(y_test, y_pred)
f1s = f1_score(y_test, y_pred, average='weighted')
print(X_train.shape[1])
print(f1s)


# In[ ]:


# In[66]:


dataset_q2 = "./dataset/IMFDB/"
# dataset_q2 =
file = dataset_q2 + "emotion.txt"

fnames = []
emot = []

with open(file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        #         print(row)
        fnames.append(row)
csv_file.close()

# print(fnames)
X_im = []
fnames = np.array(fnames)
emot = np.unique(fnames[:, 1])

emo_dict = {}
emo_rev_dict = {}
for i in range(emot.shape[0]):
    emo_dict[i] = emot[i]
    emo_rev_dict[emot[i]] = i
# print(emo_dict)
# print(emo_rev_dict)

y_emot = []

for ff in fnames:
    X_im_f = load_image(dataset_q2+ff[0])
    X_im.append(X_im_f)
    y_emot.append(emo_rev_dict[ff[1]])
X_im = np.array(X_im)
y_emot = np.array(y_emot)

XP = load_data(dataset_q2)
N, H, W = X_im.shape[0:3]
C = 1 if opt['is_grayscale'] else X_im.shape[3]
# C = X_im.shape[3]
X_re_im = X_im.reshape((N, H*W*C))
# print(X_re_im.shape)

dataset_q2_2 = "./dataset/Yale_face_database/"
file_2 = dataset_q2_2 + "emotion.txt"

fnames_2 = []
emot_2 = []

with open(file_2) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        #         print(row)
        fnames_2.append(row)
csv_file.close()

X_im_2 = []
fnames_2 = np.array(fnames_2)
emot_2 = np.unique(fnames_2[:, 1])

emo_dict_2 = {}
emo_rev_dict_2 = {}
for i in range(emot_2.shape[0]):
    emo_dict_2[i] = emot_2[i]
    emo_rev_dict_2[emot_2[i]] = i

y_emot_2 = []

for ff in fnames_2:
    X_im_f_2 = load_image(dataset_q2_2+ff[0])
    X_im_2.append(X_im_f_2)
    y_emot_2.append(emo_rev_dict_2[ff[1]])
X_im_2 = np.array(X_im_2)
y_emot_2 = np.array(y_emot_2)

XP_2 = load_data(dataset_q2_2)
N, H, W = X_im_2.shape[0:3]
C = 1 if opt['is_grayscale'] else X_im_2.shape[3]
X_re_im_2 = X_im_2.reshape((N, H*W*C))

#
## In[67]:


X_re_im = get_lda(X_re_im, y_emot, 20)
X_re_im_2 = get_lda(X_re_im_2, y_emot_2, 20)


# In[68]:


clf_emo = mlp_mod(solver='adam', activation='tanh', learning_rate='adaptive', max_iter=1000,
                  alpha=1e-3, epsilon=0.1, beta_1=0.2, beta_2=0.2, hidden_layer_sizes=(100, 100))


# In[69]:


best_acc = 0
best_mod = 0

kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X_re_im):
    X_train, X_test = X_re_im[train_index], X_re_im[test_index]
    y_train, y_test = y_emot[train_index], y_emot[test_index]
    clf_emo = mlp_mod(solver='adam', activation='tanh', learning_rate='adaptive', max_iter=1000,
                      learning_rate_init=1e-3, alpha=1e-3, epsilon=0.1, beta_1=0.2, beta_2=0.8, hidden_layer_sizes=(100, 100))
    clf_emo.fit(X_train, y_train)
    y_pred = clf_emo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    mis_c = np.where(y_test != y_pred)[0]
    if(acc > best_acc):
        best_acc = acc
        best_mod = clf_emo


print(accuracy_score(best_mod.predict(X_re_im), y_emot))
print("Done")
