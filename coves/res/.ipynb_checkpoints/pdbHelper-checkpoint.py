
def get_atoms_neighbor(model, target_chain, target_resn, n_neighbours = 80):
    '''
    model : biopython pdb model

    returns: 
        backbone atom coordinates of residue of interest
        atom_type and coordinates of all residues within a certain distance.

    '''

    # getting coordinates of the residue of interest
    chains = list(model.get_chains())
    chain_oi = [chain for chain in chains if chain._id == target_chain][0] # throw error if not found
    residues = list(chain_oi.get_residues())
    res_oi = [res for res in residues if res._id[1] == target_resn][0] # throw error if not found
    label_aa3 = res_oi.resname
    label_aa1 = AA3_TO_AA1[label_aa3]
    atoms_oi = list(res_oi.get_atoms())
    n_oi = [atom for atom in atoms_oi if atom.id == 'N'][0]
    ca_oi = [atom for atom in atoms_oi if atom.id == 'CA'][0]
    c_oi = [atom for atom in atoms_oi if atom.id == 'C'][0]

    n_oi_coord = n_oi.coord
    ca_oi_coord = ca_oi.coord
    c_oi_coord = c_oi.coord
    #print(n_oi_coord, ca_oi_coord, c_oi_coord)
    #print(n_oi_coord.dtype)
    #print(n_oi_coord - ca_oi_coord)

    # get coordinates of atoms in vicinity
    #make a list, sort it, and keep that order to insert new elements while iterating through all atoms
    # Todo: manual testing distances and vicinities.
    neighbours = []
    for chain in chains:
        residues = list(chain.get_residues())
        for residue in residues:
            #filter out the atoms from the target of interest.
            # filter for heteroatoms??
            if residue._id != target_resn and chain._id != target_chain:

                atoms = list(residue.get_atoms())
                for atom in atoms:
                    if atom.name[0] in PDB_ATOMS:
                        atom_coord = atom.coord
                        # calc distance
                        d = d_euclid(ca_oi_coord, atom_coord)

                        if len(neighbours) < n_neighbours:
                            neighbours.append((atom.name, atom_coord, d))
                            neighbours = sorted(neighbours, key = lambda x: x[2])

                        elif len(neighbours) == n_neighbours:
                            max_d = neighbours[-1][2]
                            if d < max_d:
                                #insert in order
                                to_insert = (atom.name, atom_coord, d)
                                insort_left(neighbours, to_insert, key = lambda x: x[2])
                                #drop the last element
                                neighbours = neighbours[:-1]
                            else:
                                continue

    return [n_oi_coord, ca_oi_coord, c_oi_coord], neighbours, label_aa1 