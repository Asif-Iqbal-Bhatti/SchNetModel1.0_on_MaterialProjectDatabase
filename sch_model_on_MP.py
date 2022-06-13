#!/usr/bin/env python

import os, sys
from schnetpack.datasets.matproj import MaterialsProject
import schnetpack as spk

os.environ['OPENBLAS_NUM_THREADS'] = '1'
matprojdata = MaterialsProject('./matproj.db', apikey = 'ABC', download=False, load_only=['formation_energy_per_atom', 'energy_per_atom'], )

print('Number of reference calculations:', len(matprojdata))
print('Available properties:')
for p in matprojdata.available_properties: print('-', p)

matprojtut = './matprojtut'
if not os.path.exists('matprojtut'):
    os.makedirs(matprojtut)
		
train, val, test = spk.train_test_split(
        data=matprojdata,
        num_train=90000,
        num_val=50000,
        split_file=os.path.join(matprojtut, "split.npz"),
    )
train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100)			

atomrefs = matprojdata.get_atomref('formation_energy_per_atom')
means, stddevs = train_loader.get_statistics(
    'formation_energy_per_atom', divide_by_atoms=True, single_atom_ref=atomrefs
)
print('Mean atomization energy / atom:', means['formation_energy_per_atom'])
print('Std. dev. atomization energy / atom:', stddevs['formation_energy_per_atom'])

######## Building the model ########

schnet = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff)

output_U0 = spk.atomistic.Atomwise(n_in=30, atomref=atomrefs['formation_energy_per_atom'], property='formation_energy_per_atom', mean=means['formation_energy_per_atom'], stddev=stddevs['formation_energy_per_atom'])

model = spk.AtomisticModel(representation=schnet, output_modules=output_U0)

######## Training the model ########

from torch.optim import Adam
# loss function
def mse_loss(batch, result):
    diff = batch['formation_energy_per_atom']-result['formation_energy_per_atom']
    err_sq = torch.mean(diff ** 2)
    return err_sq

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-2)

import schnetpack.train as trn

loss = trn.build_mse_loss(['formation_energy_per_atom'])

metrics = [spk.metrics.MeanAbsoluteError('formation_energy_per_atom')]
hooks = [
    trn.CSVHook(log_path=matprojtut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=matprojtut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

device = "cpu" # change to 'cpu' if gpu is not available
n_epochs = 900 # takes about 10 min on a notebook GPU. reduces for playing around
trainer.train(device=device, n_epochs=n_epochs)


##### PLOTING A FUNCTION #####
import numpy as np
import matplotlib.pyplot as plt
from ase.units import kcal, mol

results = np.loadtxt(os.path.join(matprojtut, 'log.csv'), skiprows=1, delimiter=',')

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

