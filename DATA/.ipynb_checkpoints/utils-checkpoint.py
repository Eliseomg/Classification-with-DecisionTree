from sklearn.externals.six import StringIO
from sklearn import tree
import matplotlib.pyplot as plt
import pydotplus
import matplotlib.image as mpimg
import numpy as np

def draw_tree(data, decision_tree, nombre, numAtributos, y_train):
    """
        nombre: el nombre del archivo png a guardar
        numAtributos: cantidad de atributos utilizados
    """
    dot_data = StringIO()
    atributos = data.columns[2:numAtributos]
    
    etiquetas = data["diagnosis"].unique().tolist()
    
    tree.export_graphviz(decision_tree,feature_names = atributos, out_file = dot_data,
                         class_names = np.unique(y_train), filled = True,  special_characters = True,rotate = False) 
    
    # Conversion
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
    #crear archivo
    graph.write_png(nombre)
    
    #mostrar imagen
    img_entropy = mpimg.imread(nombre)
    plt.figure(figsize=(100, 200))
    plt.imshow(img_entropy, interpolation='nearest')