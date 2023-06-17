

# one hot encoder for a single amino acid
alphabet = AA_LIST_ALPHABETICAL

def atom_to_oh(atom):
    atom_to_idx = dict(zip(PDB_ATOMS, list(range(len(PDB_ATOMS)))))
    oh = np.zeros(len(PDB_ATOMS))
    oh[atom_to_idx[atom]] = 1
    return oh

def oh_to_atom(oh):
    atom_idx = np.where(oh==1)[0][0]  

    idx_to_atom = dict(zip(list(range(len(PDB_ATOMS))),PDB_ATOMS))
    atom = idx_to_atom[atom_idx]
    return atom

def aa_to_idx(aa):
    aa_to_idx = dict(zip(alphabet, list(range(len(alphabet)))))
    return aa_to_idx[aa]

def idx_to_aa(idx):
    idx_to_aa = dict(zip( list(range(len(alphabet))), alphabet))
    return idx_to_aa[idx]

def aa_to_oh(aa):
    aa_to_idx = dict(zip(alphabet, list(range(len(alphabet)))))

    oh = np.zeros(len(alphabet))
    oh[aa_to_idx[aa]] = 1
    return oh

def oh_to_aa(oh):
    aa_idx = np.where(oh1==1)[0][0]  
    
    idx_to_aa = dict(zip( list(range(len(alphabet))), alphabet))

    aa = idx_to_aa[aa_idx]
    return aa

for aa in alphabet:
    oh1 = aa_to_oh(aa)
    aa_infer = oh_to_aa(oh1)
    assert aa == aa_infer

for atom in PDB_ATOMS:
    oh = atom_to_oh(atom)
    atom_infer = oh_to_atom(oh)
    print(atom, oh, atom_infer)

    
def d_euclid(x,y):
    return np.sqrt(np.sum((x-y)**2))

def length(x):
    return np.sqrt(np.sum((x)**2))

import math

def angle_2_vecs(x,y):
    return math.acos(np.dot(x,y)/(length(x) * length(y)))

d_euclid(np.array([0,0,0]), np.array([0,0.5,1]))