import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon
import networkx as nx
import torch
import torch.nn.functional as F
from skimage import transform
import matplotlib.image as mpimg
from torch.utils.data import DataLoader
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from floortrans.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
#discrete_cmap()
from floortrans.post_prosessing import split_prediction, get_polygons, split_validation
from mpl_toolkits.axes_grid1 import AxesGrid

rot = RotateNTurns() #
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
room_classes.append("Door")
data_folder = 'data/cubicasa5k/'
data_file = 'test.txt'
normal_set = FloorplanSVG(data_folder, data_file, format='txt', original_size=True)
data_loader = DataLoader(normal_set, batch_size=1, num_workers=0)
data_iter = iter(data_loader)
# Setup Model
model = get_model('hg_furukawa_original', 51)

n_classes = 44
split = [21, 12, 11]
#model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
#model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
#checkpoint = torch.load('model_best_val_loss_var.pkl')

#model.load_state_dict(checkpoint['model_state'])
#model.eval()
#model.cuda()
print("Model loaded.")

# with open('/home/alishakhan/notebooks/CubiCasa5k/data/test_modified_1.pkl', 'rb') as f:
#     test = pickle.load(f)

# with open('/home/alishakhan/notebooks/CubiCasa5k/data/val_modified_1.pkl', 'rb') as f:
#     val = pickle.load(f)

with open ("/home/alishakhan/notebooks/CubiCasa5k/data/test_modified_1.pkl", 'rb') as f:
    test=pickle.load(f)
def isolate_class(rooms, CLASS: int):
    template = np.zeros_like(rooms)
    rows, cols = np.where(rooms == CLASS)
    template[rows, cols] = 1
    return template
    
bad=[0, 1, 2, 8, 11]
good=[3,4,5,6,7,9,10,12]

def vis_nodes(img, significant_nodes):
    #signficant nodes exclude rooms we don't care about
    nodes = {}
    room_contours={}
    #door_contours={}
    for c in significant_nodes:
        nodes[c] = []
        room_contours[c] = []
        t = isolate_class(img, c)
        contours, _ = cv2.findContours(t.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        for s in contours:
            room_contours[c].append(s)
            nodes[c].append(np.squeeze(np.array(s), 1).mean(0))
    template = img.copy()
    #command k c to block comment
    # plt.figure(figsize=(10, 10))
    # plt.imshow(template)
    # for n in nodes.keys():
    #     for p in nodes[n]:
    #         plt.text(p[0], p[1], f"{room_classes[n]}")
    #         plt.scatter(p[0], p[1], alpha=0.6)
    # plt.show()
    return(room_contours, room_contours[12], nodes) #room contours, door contours

def get_edges(img, room_contours, door_contours):
    connections_int = []
    connections_vis = []
    nodes={}
    for i, room1 in enumerate(room_contours):
        if len(room1)<4:
            return(-1)
        room1_arr = np.array(room1).squeeze(1) 
        room1_ply = Polygon(room1_arr).buffer(1)
        for j, room2 in enumerate(room_contours):
            if len(room2)<4:
                return(-1)
            room2_arr = np.array(room2).squeeze(1)
            room2_ply = Polygon(room2_arr).buffer(1)
            if i != j:
                if room1_ply.intersects(room2_ply):
                    connections_int.append([i, j])
                    connections_vis.append([
                        room1_arr.mean(0),
                        room2_arr.mean(0)
                    ])
                else:
                    for door in door_contours:
                        door = Polygon(np.array(door).squeeze(1)).buffer(1)
                        if (room1_ply.intersects(door) and room2_ply.intersects(door)):
                            connections_int.append([i, j])
                            connections_vis.append([
                                room1_arr.mean(0),
                                room2_arr.mean(0)
                            ])
    return connections_int, connections_vis

#check that floorplan doesn't contain undefined
embeddings = None
Y= None
for index, floorplan in test.items():
    if 11 not in set(floorplan.flatten()):
        icons = normal_set[index]['label'][1].numpy()
        rows, column = np.where(icons == 2)
        floorplan[rows, column] = 12
        rooms, doors, nodes = vis_nodes(floorplan, good)
        rc = []
        attributes={"type":[], "areas":[]}
        
        for k in rooms.keys():
            if k != 12:
                rc += rooms[k]
                attributes['type'] += ([k] * len(rooms[k]))
                for r in rooms[k]:
                    r=np.array(r).squeeze(1)
                    attributes['areas'].append(Polygon(r).area)
        positions = []
        for k in rooms.keys():
            if k != 12:
                for cont in rooms[k]:
                    positions.append(np.array(cont).squeeze(1).mean(0).tolist())

        pos_attrs = {}
        for i, n in enumerate(positions):
            pos_attrs[i] = [n[0], -n[1]]
        try:
            idx, vis = get_edges(floorplan, rc, doors)
        except:
            get_edges(floorplan, rc, doors)==-1
            continue

        if not idx:
            continue

        #nodes_lst = []
        #for k in rooms.keys():
            #if k != 12:
                #nodes_lst += ([k] * len(rooms[k]))
        nodes_lst_updated = []
        for i in range(len(attributes["type"])):
            edges = set(np.array(idx).flatten())
            if i in edges:
                nodes_lst_updated.append(attributes["type"][i])
        #nodes_lst = nodes_lst_updated

        node_attrs = {}
        for i, n in enumerate(nodes_lst_updated):
            node_attrs[i] = room_classes[n]

        G = nx.Graph(idx)
        A = nx.adjacency_matrix(G)
        X = F.one_hot(torch.tensor(nodes_lst_updated), 11).numpy()
        try:
            H = A @ X
        except:
            print('ERROR')
            print(A.shape)
            print(X.shape)
            break
        if embeddings is None:
            embeddings=H
        else:
            embeddings = np.concatenate(([embeddings , H ]), axis=0)
        
        if Y is None:
            Y=X
        else:
            Y=np.concatenate(([Y, X]), axis=0)


# data_file = 'val.txt'
# normal_set = FloorplanSVG(data_folder, data_file, format='txt', original_size=True)
# data_loader = DataLoader(normal_set, batch_size=1, num_workers=0)
# data_iter = iter(data_loader)
# # Setup Model
# model = get_model('hg_furukawa_original', 51)

# for index, floorplan in val.items():
#     if 11 not in set(floorplan.flatten()):
#         icons = normal_set[index]['label'][1].numpy()
#         rows, column = np.where(icons == 2)
#         floorplan[rows, column] = 12
#         rooms, doors, nodes = vis_nodes(floorplan, good)
#         rc = []
#         for k in rooms.keys():
#             if k != 12:
#                 rc += rooms[k]

#         positions = []
#         for k in rooms.keys():
#             if k != 12:
#                 for cont in rooms[k]:
#                     positions.append(np.array(cont).squeeze(1).mean(0).tolist())

#         pos_attrs = {}
#         for i, n in enumerate(positions):
#             pos_attrs[i] = [n[0], -n[1]]
#         try:
#             idx, vis = get_edges(floorplan, rc, doors)
#         except:
#             get_edges(floorplan, rc, doors)==-1
#             continue

#         if not idx:
#             continue

#         nodes_lst = []
#         for k in rooms.keys():
#             if k != 12:
#                 nodes_lst += ([k] * len(rooms[k]))
#         nodes_lst_updated = []
#         for i in range(len(nodes_lst)):
#             edges = set(np.array(idx).flatten())
#             if i in edges:
#                 nodes_lst_updated.append(nodes_lst[i])
#         nodes_lst = nodes_lst_updated

#         node_attrs = {}
#         for i, n in enumerate(nodes_lst):
#             node_attrs[i] = room_classes[n]

#         G = nx.Graph(idx)
#         A = nx.adjacency_matrix(G)
#         X = F.one_hot(torch.tensor(nodes_lst), 11).numpy()
#         try:
#             H = A @ X
#         except:
#             print('ERROR')
#             print(A.shape)
#             print(X.shape)
#             break
        
#         embeddings = np.concatenate(([embeddings , H ]), axis=0)
#         Y=np.concatenate(([Y, X]), axis=0)

np.save("/home/alishakhan/notebooks/CubiCasa5k/outputs/embeddings_test_trial.npy", embeddings)
np.save("/home/alishakhan/notebooks/CubiCasa5k/outputs/Y_test_trial.npy", Y)