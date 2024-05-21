import os
import torch

print('[3-2 build_fullyconnected_pth.py]')
w = torch.load('./best_gnn.pth')
del w['conv1.lin_l.weight']
del w['conv1.lin_l.bias']
del w['conv1.lin_r.weight']
del w['conv2.lin_l.weight']
del w['conv2.lin_l.bias']
del w['conv2.lin_r.weight']
del w['fc1.weight']
del w['fc1.bias']
torch.save(w, './fully_connected.pth')