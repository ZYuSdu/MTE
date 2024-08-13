import argparse
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, f1_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

dataPrePath = 'evalData/'
house_zone_index = dataPrePath + 'sz_housePrice_index.txt'
house_zone_price = dataPrePath + 'sz_housePrice.txt'
pop_zone_index2020 = dataPrePath + 'sz_population_index.txt'
pop_zone2020 = dataPrePath + 'sz_population.txt'
zone_basePath = dataPrePath + 'baseStation2Zone.txt'

baselinePrePath = 'zone_embed/'
mte_different_temporal_embed_path = baselinePrePath + 'mte_temporal_embed.tensor'
mte_different_spatial_embed_path = baselinePrePath + 'mte_spatial_embed.tensor'



def classification(zone_NoFre,savePointNumber):
    zoneNum = zone_NoFre.shape[0]
    X = zone_NoFre
    y = np.random.randint(0,7, zoneNum)
    res = []
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1 / 3)
    clf = RandomForestClassifier(max_depth=None, min_samples_split=2)
    clf.fit(train_X, train_y)
    pridict_y = clf.predict(test_X)
    acc = accuracy_score(test_y, pridict_y)
    res.append(round(acc, savePointNumber))

    pre_weighted = precision_score(test_y, pridict_y, average='weighted')
    res.append(round(pre_weighted, savePointNumber))

    f1_weighted = f1_score(test_y, pridict_y, average='weighted')
    res.append(round(f1_weighted, savePointNumber))

    kappa = cohen_kappa_score(test_y, pridict_y)
    res.append(round(kappa, savePointNumber))

    return res


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def house_regression(zone_NoFre,savePointNumber):
    truth_index = np.loadtxt(house_zone_index, dtype=int)
    X = zone_NoFre[truth_index]
    y = np.loadtxt(house_zone_price)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1 / 3)
    clf = RandomForestRegressor(max_depth=None, min_samples_split=2)
    clf.fit(train_X, train_y)
    res = []
    test_score = clf.score(test_X, test_y)
    res.append(round(test_score, savePointNumber))
    pred_y = clf.predict(test_X)

    RMSE = np.sqrt(mean_squared_error(test_y, pred_y))
    res.append(round(RMSE, savePointNumber))

    MAE = mean_absolute_error(test_y, pred_y)
    res.append(round(MAE, savePointNumber))

    MAPE = mape(test_y, pred_y)
    res.append(round(MAPE, savePointNumber))
    return res


def population_regression(zone_NoFre,savePointNumber):
    truth_index = np.loadtxt(pop_zone_index2020, dtype=int)
    X = zone_NoFre[truth_index]
    y = np.loadtxt(pop_zone2020)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1 / 3)
    clf = RandomForestRegressor(max_depth=None, min_samples_split=2)
    clf.fit(train_X, train_y)
    res = []
    test_score = clf.score(test_X, test_y)
    res.append(round(test_score, savePointNumber))

    pred_y = clf.predict(test_X)

    RMSE = np.sqrt(mean_squared_error(test_y, pred_y))
    res.append(round(RMSE, savePointNumber))

    MAE = mean_absolute_error(test_y, pred_y)
    res.append(round(MAE, savePointNumber))

    MAPE = mape(test_y, pred_y)
    res.append(round(MAPE, savePointNumber))
    return res


def evalZoneEmbed(zone_NoFre, epoch,savePointNumber):
    res = [epoch]
    res_class = classification(zone_NoFre,savePointNumber)  # "acc","weighted-pre","weighted-recall","weighted-f1","kappa"
    res = res + res_class
    res_reg = house_regression(zone_NoFre,savePointNumber)  # "test-r2","mnse","mae","mape"
    res = res + res_reg
    res_reg = population_regression(zone_NoFre,savePointNumber)  # "test-r2","mnse","mae","mape"
    res = res + res_reg
    return res




def evalbaseline(eval_embed_path,args ):
    zoneEmbed = torch.load(eval_embed_path).cpu().numpy()
    for iter in range(0, args.epochs):
        line = evalZoneEmbed(zoneEmbed, str(iter),args.fractionalDigits)
        print(line)


def evalMTE(mte_temporal_embed, mte_spatial_embed, args):
    temporal_embed = torch.load(mte_temporal_embed).cpu().numpy()
    spatial_embed = torch.load(mte_spatial_embed).cpu().numpy()
    zoneEmbed = np.c_[temporal_embed, spatial_embed]
    for iter in range(0, args.epochs):
        line = evalZoneEmbed(zoneEmbed, str(iter),args.fractionalDigits)
        print(line)


def parse_args():
    """ parsing the arguments that are used in HGI """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fractionalDigits', type=int, default=3)
    parser.add_argument('--task', type=str, default='mte_spatial', help='mte|mte_spatial|mte_temporal|skipgram3view|graph3view|hdge|hier|mne')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    task = args.task
    print(task)
    print("      |  land use classification  | house price prediction | population density prediction")
    title = "iter, |Accuracy, Precision, F1, kappa,| R2, RMSE, MAE, MAPE,| R2, RMSE, MAE, MAPE"
    print(title)
    if task == 'mte':
        evalMTE(mte_different_temporal_embed_path, mte_different_spatial_embed_path, args)
    elif task == 'mte_spatial':
        evalbaseline(mte_different_spatial_embed_path, args)
    elif task == 'mte_temporal':
        evalbaseline(mte_different_temporal_embed_path, args)

