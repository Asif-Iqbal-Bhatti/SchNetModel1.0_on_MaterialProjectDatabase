#!/usr/bin/env python3
	
'''
#========================================================
# AUTHOR:: AsifIqbal -> @AIB_EM
# USAGE :: Training a material project database
#       :: schnetpack download from conda
#========================================================
'''

import os, sys, numpy as np, matplotlib.pyplot as plt
import schnetpack as spk
import schnetpack.train as trn
from ase.units import kcal, mol
import torch, torchmetrics
from torch.optim import Adam
import pytorch_lightning as pl
from schnetpack.datasets.matproj import MaterialsProject

os.environ['OPENBLAS_NUM_THREADS'] = '1'
device = "gpu" # change to 'cpu' if gpu is not available
properMP = 'formation_energy_per_atom'
outData = './mpModelData'
if not os.path.exists('outData'):   
    os.makedirs(outData, exist_ok=True)

#============ INITIALIZATION
def data_loading():
    key="***********"
    matprojdata = MaterialsProject('./matprojAI.db', apikey=key, download=False,)
    print('Total calculations:', len(matprojdata))
    print('Available properties:')
    for p in matprojdata.available_properties: 
        print('->', p)
    train, val, test = spk.train_test_split(
        data=matprojdata,
        num_train=80000,
        num_val=40000,
        split_file=os.path.join(outData, "split.npz"),)
    train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
    val_loader = spk.AtomsLoader(val, batch_size=100)
    
    atomrefs = matprojdata.get_atomref(properMP)

    means, stddevs = train_loader.get_statistics(properMP, divide_by_atoms=True, single_atom_ref=atomrefs)
    print('Mean For_energy/atom:', means[properMP])
    print('Std. dev. For_energy/atom:', stddevs[properMP])
    return atomrefs, test, means, stddevs, train_loader, val_loader
    
#============ LOSS FUNCTION
def mse_loss(batch, result):
	diff = batch[properMP]-result[properMP]
	return torch.mean(diff ** 2)

#============ BUILDING THE MODEL 
def schnet_model(atomrefs, means, stddevs, train_loader, val_loader):
    schnet = spk.representation.SchNet(
        n_atom_basis=30, 
        n_filters=30, 
        n_gaussians=20, 
        n_interactions=5,
        cutoff=4., 
        cutoff_network=spk.nn.cutoff.CosineCutoff)
    output_U0=spk.atomistic.Atomwise(
    n_in=30, 
    atomref=atomrefs[properMP], 
    property=properMP, 
    mean=means[properMP], 
    stddev=stddevs[properMP]
    )
    model = spk.AtomisticModel(representation=schnet, output_modules=output_U0)

    #============ TRAINING THE MODEL 
    optimizer = Adam(model.parameters(), lr=0.01)
    loss = trn.build_mse_loss([properMP])
    
    metrics = [spk.metrics.MeanAbsoluteError(properMP)]
    hooks = [
        trn.CSVHook(log_path=outData, metrics=metrics),
        trn.ReduceLROnPlateauHook(
            optimizer,
            patience=5, factor=0.8, min_lr=1e-4,
            stop_after_min=True)
    ]

    trainer = trn.Trainer(
        model_path=outData,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    n_epochs = 5
    trainer.train(device=device, n_epochs=n_epochs)

#============ PLOTING A FUNCTION 
def result_plot():
    results = np.loadtxt(os.path.join(outData, 'log.csv'), skiprows=1, delimiter=',')
    
    time = results[:,0]-results[0,0]
    learning_rate = results[:,1]
    train_loss = results[:,2]
    val_loss = results[:,3]
    val_mae = results[:,4]
    
    print('Final validation MAE:', np.round(val_mae[-1], 2), 'eV =',
        np.round(val_mae[-1] / (kcal/mol), 2), 'kcal/mol')
    
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(time, val_loss, label='Validation')
    plt.plot(time, train_loss, label='Train')
    plt.yscale('log')
    plt.ylabel('Loss [eV]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(time, val_mae)
    plt.ylabel('mean abs. error [eV]')
    plt.xlabel('Time [s]')
    plt.savefig('matproj.png')

#============ USING THE MODEL FOR PREDICTION
def model_pred(test):
    best_model = torch.load(os.path.join(outData, 'best_model'))
    test_loader = spk.AtomsLoader(test, batch_size=100)
    
    err = 0
    print(len(test_loader))
    for count, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        # apply model
        pred = best_model(batch)
        # calculate absolute error
        tmp = torch.sum(torch.abs(pred[properMP]-batch[properMP]))
        tmp = tmp.detach().cpu().numpy() # detach from graph & convert to numpy
        err += tmp
    
        # log progress
        percent = '{:3.2f}'.format(count/len(test_loader)*100)
        print('Progress:', f'{percent}%' + ' '*(5-len(percent)), end="\r")
    
    err /= len(test)
    print('Test MAE', np.round(err, 2), 'eV =',
        np.round(err / (kcal/mol), 2), 'kcal/mol')
			

atomrefs, test, means, stddevs, train_loader, val_loader = data_loading()
schnet_model(atomrefs, means, stddevs, train_loader, val_loader)
result_plot()
model_pred(test)
