

"""Bisection algorithms."""
def insort_right(a, x, lo=0, hi=None, *, key=None):
    """Insert item x in list a, and keep it sorted assuming a is sorted.
    If x is already in a, insert it to the right of the rightmost x.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if key is None:
        lo = bisect_right(a, x, lo, hi)
    else:
        lo = bisect_right(a, key(x), lo, hi, key=key)
    a.insert(lo, x)


def bisect_right(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo


def insort_left(a, x, lo=0, hi=None, *, key=None):
    """Insert item x in list a, and keep it sorted assuming a is sorted.
    If x is already in a, insert it to the left of the leftmost x.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if key is None:
        lo = bisect_left(a, x, lo, hi)
    else:
        lo = bisect_left(a, key(x), lo, hi, key=key)
    a.insert(lo, x)

def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo
#################################################################################################################
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

def proj(v1, b):
    #project vector v1 onto basis vector b
    b_norm = np.sqrt(sum(b**2))
    proj = (np.dot(v1, b)/b_norm**2) * b
    return proj

def get_stand_coords(n, ca, c, x):
    ''' 
    express the x coordinate in terms of the reference frame of the basis 
    spanned by ca-n, ca-c, and the crossproduct here.
    #make unit vectors of the basis
    then project the dfference vector x-ca onto each of these unit basis vectors

    try: 
        - smooth out the angle between the N-CA CA-C vectors to 120.
        - smooth out the angle between the place spanned by the peptide bond and the CB
    '''

    n_ca = n - ca
    u_n_ca = n_ca /length(n_ca) # unit vector
    #print(u_n_ca, length(u_n_ca))
    c_ca = c - ca
    u_c_ca = c_ca / length(c_ca) # unit vector 2

    cross = np.cross(u_n_ca, u_c_ca)
    u_cross = cross /length(cross)
    #print(length(u_cross))

    # check cross product is orthogonal
    #print(np.dot(u_cross, u_c_ca))
    #print(np.dot(u_cross, u_ca_n))


    x_ca = x- ca

    ###### find the new coordinates in the basis spanned by the peptide backbone.
    # inverse takes a while.
    #starttime = timeit.default_timer()
    #vec_new = np.linalg.inv(np.array([u_ca_n, u_c_ca, u_cross])).dot(x_ca)    
    # or: np.linalg.solve(W, v)
    #print(vec_new)
    #endtime = timeit.default_timer()
    #print(endtime - starttime)

    #starttime = timeit.default_timer()
    vec_new = np.linalg.solve(np.array([u_n_ca, u_c_ca, u_cross]), x_ca)
    #print(vec_new)
    #endtime = timeit.default_timer()
    #print(endtime - starttime)


    return vec_new