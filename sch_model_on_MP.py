#!/usr/bin/env python
	
'''
#
# AUTHOR:: AsifIqbal -> @AIB_EM
# USAGE :: 
#
'''

import os, sys
from schnetpack.datasets.matproj import MaterialsProject
import schnetpack as spk
from torch.optim import Adam
import schnetpack.train as trn

import numpy as np
import matplotlib.pyplot as plt
from ase.units import kcal, mol
import torch

#==== INITIALIZATION
os.environ['OPENBLAS_NUM_THREADS'] = '1'
key='NAN'
matprojdata = MaterialsProject('./matprojAI.db', apikey = key, download=False, load_only=['formation_energy_per_atom', 'energy_per_atom'], )
outData = './mpModelData'
if not os.path.exists('outData'):
	os.makedirs(outData)
#====

print('Total calculations:', len(matprojdata))
print('Available properties:')
for p in matprojdata.available_properties: print('-', p)

# loss function
def mse_loss(batch, result):
    diff = batch['formation_energy_per_atom']-result['formation_energy_per_atom']
    err_sq = torch.mean(diff ** 2)
    return err_sq
		
train, val, test = spk.train_test_split(
		data=matprojdata,
		num_train=80000,
		num_val=40000,
		split_file=os.path.join(outData, "split.npz"),
		)
		
train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100)			

atomrefs = matprojdata.get_atomref('formation_energy_per_atom')
print(atomrefs)

means, stddevs = train_loader.get_statistics(
    'formation_energy_per_atom', divide_by_atoms=True, single_atom_ref=atomrefs
)
print('Mean For_energy/atom:', means['formation_energy_per_atom'])
print('Std. dev. For_energy/atom:', stddevs['formation_energy_per_atom'])

#==== Building the model 
schnet = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff)
output_U0 = spk.atomistic.Atomwise(n_in=30, atomref=atomrefs['formation_energy_per_atom'], property='formation_energy_per_atom', mean=means['formation_energy_per_atom'], stddev=stddevs['formation_energy_per_atom'])
model = spk.AtomisticModel(representation=schnet, output_modules=output_U0)

#==== Training the model 
# build optimizer
optimizer = Adam(model.parameters(), lr=1e-2)
loss = trn.build_mse_loss(['formation_energy_per_atom'])

metrics = [spk.metrics.MeanAbsoluteError('formation_energy_per_atom')]
hooks = [
    trn.CSVHook(log_path=outData, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
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

device = "cpu" # change to 'cpu' if gpu is not available
n_epochs = 200 
trainer.train(device=device, n_epochs=n_epochs)

#==== PLOTING A FUNCTION 
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
plt.show()

#==== Using the model for prediction
best_model = torch.load(os.path.join(outData, 'best_model'))
test_loader = spk.AtomsLoader(test, batch_size=100)

err = 0
print(len(test_loader))
for count, batch in enumerate(test_loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    # apply model
    pred = best_model(batch)
    # calculate absolute error
    tmp = torch.sum(torch.abs(pred['formation_energy_per_atom']-batch['formation_energy_per_atom']))
    tmp = tmp.detach().cpu().numpy() # detach from graph & convert to numpy
    err += tmp

    # log progress
    percent = '{:3.2f}'.format(count/len(test_loader)*100)
    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

err /= len(test)
print('Test MAE', np.round(err, 2), 'eV =',
	np.round(err / (kcal/mol), 2), 'kcal/mol')
			
