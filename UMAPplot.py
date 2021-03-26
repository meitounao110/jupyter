import torch
import numpy as np
import math
# import matplotlib.pyplot as plt
# import umap
# from util import datasets


# set = datasets.dataloader(root='/mnt/zhaoxiaoyu/data/layout_data/simple_component/one_point_dataset_200x200/train/',
#                           list_path='/mnt/zhangyunyang1/pseudo_label-pytorch-master/100label.txt',
#                           max_iters=None)
# loader = torch.utils.data.DataLoader(set,
#                                      batch_size=25,
#                                      shuffle=False,
#                                      num_workers=0,
#                                      pin_memory=True,
#                                      drop_last=False)
# # images1 = []
# # images2 = []
# # datas = []
# for idx, (data, targets) in enumerate(loader):
#     data = data.numpy()
#     targets = targets.numpy()
#     data1 = np.reshape(data, (25, -1))
#     targets1 = np.reshape(targets, (25, -1))
#     reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="correlation")
#     embedding = reducer.fit_transform(data1)
#     embedding1 = reducer.fit_transform(targets1)
#     print(embedding.shape)
#
#     plt.figure(1)
#     plt.scatter(embedding[:, 0], embedding[:, 1], c="r", cmap='Spectral', s=20)
#     n = np.arange(25)
#     for i, txt in enumerate(n):
#         plt.annotate(txt, (embedding[:, 0][i], embedding[:, 1][i]))
#     # plt.gca().set_aspect('equal', 'datalim')
#     # plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
#     plt.title('UMAP projection of the Digits dataset')
#     plt.show()
#
#     plt.figure(2)
#     plt.scatter(embedding1[:, 0], embedding1[:, 1], c="r", cmap='Spectral', s=20)
#     n = np.arange(25)
#     for i, txt in enumerate(n):
#         plt.annotate(txt, (embedding1[:, 0][i], embedding1[:, 1][i]))
#     # plt.gca().set_aspect('equal', 'datalim')
#     # plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
#     plt.title('UMAP projection of the Digits dataset')
#     plt.show()
#     break

# digits = load_digits()
# fig, ax_array = plt.subplots(10, 10)
# axes = ax_array.flatten()
# for i, ax in enumerate(axes):
#     ax.imshow(data[i])
# plt.setp(axes, xticks=[], yticks=[], frame_on=False)
# plt.tight_layout(h_pad=0.5, w_pad=0.01)
# plt.show()





