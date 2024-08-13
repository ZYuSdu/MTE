import argparse
import scipy.sparse as sp
import torch
import numpy as np
from utils import compute_ppr, sparse_mx_to_torch_sparse_tensor
from Module.MTE import MTE
import time
import torch.nn as nn
import warnings
import os

warnings.filterwarnings("ignore")


def parse_args():
    """ parsing the arguments that are used in HGI """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=64, help='Dimension of output representation')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hid_units', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--traj2vecEmbeddingPath', type=str, default='zone_embed/traj2vec_embed.tensor')
    parser.add_argument('--temporal_graph_path', type=str, default='data/temporal_adj.npy')
    parser.add_argument('--spatial_graph_path', type=str, default='data/spatial_adj.npy')
    parser.add_argument('--temporal_embed_path', type=str, default='zone_embed/mte_temporal_embed_sample.tensor')
    parser.add_argument('--spatial_embed_path', type=str, default='zone_embed/mte_spatial_embed_sample.tensor')
    parser.add_argument('--view_name', type=str, default='temporal', help='spatial|temporal')
    return parser.parse_args()


def train(args, adjPath, TrainType, verbose=True):
    patience = args.patience
    lr = args.lr
    hid_units = args.hid_units
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    ft_size = args.dim
    featuresPath = args.traj2vecEmbeddingPath

    adj = np.load(adjPath)
    if os.path.exists('data/diff_{}.npy'.format(args.view_name)):
        diff = np.load('data/diff_{}.npy'.format(args.view_name), allow_pickle=True)
    else:
        diff = compute_ppr(adj, 0.2)
        np.save('data/diff_{}.npy'.format(args.view_name), diff)

    features = torch.load(featuresPath).numpy()[0:100,:]
    sample_size = adj.shape[0]
    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = MTE(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    model.to(args.device)
    lbl = lbl.to(args.device)

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)


        ba = torch.FloatTensor(ba)
        bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]

        if torch.cuda.is_available():
            bf = bf.to(args.device)
            ba = ba.to(args.device)
            bd = bd.to(args.device)
            shuf_fts = shuf_fts.to(args.device)

        model.train()
        optimiser.zero_grad()

        logits, __, __ = model(bf, shuf_fts, ba, bd, None, None, None)

        loss = b_xent(logits, lbl)

        loss.backward()
        optimiser.step()

        if verbose:
            if epoch % args.eval_every == 0:
                print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1



    if verbose:
        print('Loading {}th epoch, which has the lowest loss.'.format(best_t))
    model.load_state_dict(torch.load('model.pkl'))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.to(args.device)
    adj = adj.to(args.device)
    diff = diff.to(args.device)

    embeds, _ = model.embed(features, adj, diff, None)
    embed_task = torch.squeeze(embeds, axis=0)
    embed = embed_task.detach().cpu().numpy()
    baseStation2ZoneMatrix = np.load('./evalData/baseStation2Zone_sample.npy')
    zone_embed= np.dot(baseStation2ZoneMatrix.T, embed)
    torch.save(torch.from_numpy(zone_embed), TrainType)


if __name__ == '__main__':
    args = parse_args()
    starttime = time.time()
    if args.view_name == 'spatial':
        print('start train spatial view')
        train(args, args.spatial_graph_path, args.spatial_embed_path)
    else:
        print('start train temporal view')
        train(args, args.temporal_graph_path, args.temporal_embed_path)
    endtime = time.time()
    print('run time:', round(endtime - starttime, 2), 'secs')
