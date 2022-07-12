#!/usr/bin/env python3
	
'''
#===================================================================
# AUTHOR:: AsifIqbal --> @AIB_EM
# USAGE :: CREATE ASE database with Atoms object 
#          CORRECTING EHULL FOR WORKFLOW
#===================================================================

'''
	
import subprocess, numpy, json, os
from pymatgen.ext.matproj import MPRester
from monty.serialization import loadfn
from ase import Atoms
from ase.db import connect
from ase.io import read
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import numpy as np

cwd = os.path.abspath(os.path.dirname(__file__))
outdbpathF = os.path.join(cwd, 'EHCorrForces2M.db')
outFile2 = open('EHCorr.csv', 'w')

#============= DEFINE PROPERTIES
EformPerAtom = "FormationEnergy/atom"
EPerAtom     = "TotalEnergy/atom"
CEPerAtom    = "CorrTotalEnergy/atom"
EHull        = "Ehull"
Umin         = "Umin"
Umax         = "Umax"
EHullDiff    = "EhullDiff"
IonStep      = "IonicStep"
WS           = "WorkflowStep"
Dir          = "Dir"
Subdir       = "Subdir"

#============= INITIALIZATION
data_cnt = 0; wf1 = 1
inp_Argo = 'DDDD'
initial_path = os.getcwd()
abs_path = os.path.abspath(os.path.join(initial_path, inp_Argo))
argo_Dir = next(os.walk(abs_path))[1]
print('2M_ARG:: ', str(len(argo_Dir)))

#============= EVOLUTION OF WORKFLOW
dLDATA = ['A', 'B', 'C']

outFile2.write("{},{},{},{},{},{},{}\n".format("WorkflowStep", "Dir", \
	"Subdir", "IonicStep", "TotalEnergy/atom", "CorrTotalEnergy/atom", "ehull"))

with connect(outdbpathF) as conDB:
	for p in tqdm(argo_Dir):
		try:
			nn = []; ee = []; ff = {}; ll = {}
			oo = 1; asd = []; dife = []; dif4 = []
			
			for dl in dLDATA:
				arg_path  = os.path.join(inp_Argo, p, dl)
				vasp_xml	= os.path.join(arg_path,'vasprun.xml')
				read_xmlF = read(vasp_xml, index=':') # FINAL/ALL GEOMTERY (USE "-1" or ":")
				
				# WE OVERRIDE PYTHON BY INDEXING WITH 1!!!
				for ionstep, at in enumerate(read_xmlF, start=1):	
					nn.append("{}".format( ionstep ))
					ee.append("{}".format( at.get_total_energy()/at.get_global_number_of_atoms() ))
			
			# REVERSE THE DATA FOR IDENTIFY THE TAG
			nn = [int(i) for i in list(reversed(nn)) ]
			ee = [float(i) for i in list(reversed(ee)) ]
			hh = list(zip(nn,ee)); 
			
			# ANCHOR ON DIFFERENT DATA SETS!!! 
			# ALL DATA NEEDS TO BE READ IN A WORKFLOW
			for k, j in enumerate(hh, start=1):
				if j[0] == 1:
					index = k;
					temp = j[1]
					for i, pp in enumerate(hh, start=1):			
						if i == index+1:
							temp1 = pp[1]
					diff = temp-temp1	
					ff[k] = diff, temp, temp1
			#print(ff)
			
			# WE ARE INTERESTED IN THE LAST CALCULATED EHULL ONLY!!! 
			# THE LAST ONE IS ENOUGH
			read_EH = loadfn(os.path.join('workflow_data',dLDATA[2]+'.json'))
			for i, x in enumerate(read_EH):
				for k1, v1 in read_EH[i].items():
					if v1 == p:
						k2 = x['Ehull']
			#print(p, k2)
			
			#=====================================================	
			#                        MAIN
			#=====================================================
			
			for dl in reversed(dLDATA):
				arg_path  = os.path.join(inp_Argo, p, dl)
				vasp_xml	= os.path.join(arg_path,'vasprun.xml')
				read_xmlF = read(vasp_xml, index=':') # FINAL/ALL GEOMTERY (USE "-1" or ":")	
					
				# COUNT IONIC STEPS! & REVERSE IT
				for ionstep, at in enumerate(read_xmlF, start = 1):	
					ll[ionstep] = at.get_total_energy()/at.get_global_number_of_atoms()
			
				# FROM LAST TO FIRST ORDER!!!
				rr = OrderedDict(reversed(list(ll.items())));
				kk = list(rr.keys());
				vv = list(rr.values());
				key_list = list(ff.keys());
				val_list = list(ff.values());
				vvdiff = [i-vv[0] for i in vv];
					
				for u, at in enumerate(read_xmlF, start = 1):
					if oo <=  key_list[0]:
						conDB.write(at, IonStep=kk[u-1], EPerAtom=vv[u-1], Ehull=k2+vvdiff[u-1],
						data={WS:wf1, Dir:p, Subdir:dl, IonStep:kk[u-1], EPerAtom:vv[u-1], CEPerAtom:vv[u-1], EHull:k2+vvdiff[u-1] })			
						
						dif1 = k2+vvdiff[u-1]
						outFile2.write("{}, {}, {}, {}, {}, {}, {}\n". \
						format(wf1, p, dl, kk[u-1], vv[u-1], vv[u-1], k2+vvdiff[u-1]  ))
					
					if oo > key_list[0] and oo <= key_list[1]:
						asd.append(vv[u-1]+val_list[0][0])
						dif2 = [i-asd[0] for i in asd] 
						dif3 = dif1+dif2[u-1]
						
						conDB.write(at, IonStep=kk[u-1], EPerAtom=vv[u-1], Ehull=dif1+dif2[u-1] , 
						data={WS:wf1, Dir:p, Subdir:dl, IonStep:kk[u-1], EPerAtom:vv[u-1], CEPerAtom:vv[u-1]+val_list[0][0], EHull:dif1+dif2[u-1] })
						
						outFile2.write("{}, {}, {}, {}, {}, {}, {}\n". \
						format(wf1, p, dl, kk[u-1], vv[u-1], vv[u-1]+val_list[0][0], dif1+dif2[u-1]    ))
				
					if oo > key_list[1]:
						dife.append( vv[u-1]-asd[-1] )
						dif4 = [i-dife[0] for i in dife]
						
				for u, at in enumerate(read_xmlF, start = 1):
					if oo > key_list[1]:	
					
						conDB.write(at, IonStep=kk[u-1], EPerAtom=vv[u-1], Ehull=dif3+dif4[u-1],
						data={WS:wf1, Dir:p, Subdir:dl, IonStep:kk[u-1], EPerAtom:vv[u-1], CEPerAtom:vv[u-1]-dife[0], EHull:dif3+dif4[u-1]  })
						
						outFile2.write("{}, {}, {}, {}, {}, {}, {}\n". \
						format(wf1, p, dl, kk[u-1], vv[u-1], vv[u-1]-dife[0], dif3+dif4[u-1]     ))		
						
					data_cnt +=1
					oo +=1
			wf1 +=1
		except:
			pass
print("Total_Entries_in_db:: {}".format(data_cnt))
print("Total_workflow_in_db:: {}".format(wf1-1))

df = pd.read_csv('EHCorr.csv')
print (df[["IonicStep", "TotalEnergy/atom", "ehull"]])
