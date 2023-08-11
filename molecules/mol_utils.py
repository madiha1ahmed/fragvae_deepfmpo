#from global_parameters import MOL_SPLIT_START, MAX_FREE, MAX_ATOMS, MAX_FRAGMENTS
from rdkit import Chem
import numpy as np
MOL_SPLIT_START=70

# Main module for handleing the interactions with molecules





# Atom numbers of noble gases (should not be used as dummy atoms)
NOBLE_GASES = set([2, 10, 18, 36, 54, 86])
ng_correction = set()


# Drop salt from SMILES string
def drop_salt(s):
    s = s.split(".")
    return [x for _, x in sorted(zip(map(len,s), s), reverse=True)][0]




# Check if it is ok to break a bond.
# It is ok to break a bond if:
#    1. It is a single bond
#    2. Either the start or the end atom is in a ring, but not both of them.
def okToBreak(bond):

    if bond.IsInRing():
        return False

    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False


    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    if not(begin_atom.IsInRing() or end_atom.IsInRing()):
        return False
    elif begin_atom.GetAtomicNum() >= MOL_SPLIT_START or \
            end_atom.GetAtomicNum() >= MOL_SPLIT_START:
        return False
    else:
        return True



# Divide a molecule into fragments
def fragment_iterative(mol):

    split_id = MOL_SPLIT_START

    res = []
    to_check = [mol]
    while len(to_check) > 0:
        ms = spf(to_check.pop(), split_id)
        if len(ms) == 1:
            res += ms
        else:
            to_check += ms
            split_id += 1

    return create_chain(res)


# Function for doing all the nitty gritty splitting work.
def spf(mol, split_id):

    bonds = mol.GetBonds()
    for i in range(len(bonds)):
        if okToBreak(bonds[i]):
            mol = Chem.FragmentOnBonds(mol, [i], addDummies=True, dummyLabels=[(0, 0)])
            # Dummy atoms are always added last
            n_at = mol.GetNumAtoms()
            mol.GetAtomWithIdx(n_at-1).SetAtomicNum(split_id)
            mol.GetAtomWithIdx(n_at-2).SetAtomicNum(split_id)
            return Chem.rdmolops.GetMolFrags(mol, asMols=True)

    # If the molecule could not been split, return original molecule
    return [mol]



# Build up a chain of fragments from a molecule.
# This is required so that a given list of fragments can be rebuilt into the same
#   molecule as was given when splitting the molecule
def create_chain(splits):
    splits_ids = np.asarray(
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits])

    splits_ids = \
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits]

    splits2 = []
    mv = np.max(splits_ids)
    look_for = [mv if isinstance(mv, np.int64) else mv[0]]
    join_order = []

    mols = []

    for i in range(len(splits_ids)):
        l = splits_ids[i]
        if l[0] == look_for[0] and len(l) == 1:
            mols.append(splits[i])
            splits2.append(splits_ids[i])
            splits_ids[i] = []


    while len(look_for) > 0:
        sid = look_for.pop()
        join_order.append(sid)
        next_mol = [i for i in range(len(splits_ids))
                      if sid in splits_ids[i]]

        if len(next_mol) == 0:
            break
        next_mol = next_mol[0]

        for n in splits_ids[next_mol]:
            if n != sid:
                look_for.append(n)
        mols.append(splits[next_mol])
        splits2.append(splits_ids[next_mol])
        splits_ids[next_mol] = []

    return [simplify_splits(mols[i], splits2[i], join_order) for i in range(len(mols))]



# Split and keep track of the order on how to rebuild the molecule
def simplify_splits(mol, splits, join_order):

    td = {}
    n = 0
    for i in splits:
        for j in join_order:
            if i == j:
                td[i] = MOL_SPLIT_START + n
                n += 1
                if n in NOBLE_GASES:
                    n += 1


    for a in mol.GetAtoms():
        k = a.GetAtomicNum()
        if k in td:
            a.SetAtomicNum(td[k])

    return mol


# Go through a molecule and find attachment points and define in which order they should be re-joined.
def get_join_list(mol):

    join = []
    rem = []
    bonds = []

    for a in mol.GetAtoms():
        an = a.GetAtomicNum()
        if an >= MOL_SPLIT_START:
            while len(join) <= (an - MOL_SPLIT_START):
                rem.append(None)
                bonds.append(None)
                join.append(None)

            b = a.GetBonds()[0]
            ja = b.GetBeginAtom() if b.GetBeginAtom().GetAtomicNum() < MOL_SPLIT_START else \
                 b.GetEndAtom()
            join[an - MOL_SPLIT_START] = ja.GetIdx()
            rem[an - MOL_SPLIT_START] = a.GetIdx()
            bonds[an - MOL_SPLIT_START] = b.GetBondType()
            a.SetAtomicNum(0)

    return [x for x in join if x is not None],\
           [x for x in bonds if x is not None],\
           [x for x in rem if x is not None]


# Join a list of fragments toghether into a molecule
#   Throws an exception if it is not possible to join all fragments.
def join_fragments(fragments):

    to_join = []
    bonds = []
    pairs = []
    del_atoms = []
    new_mol = fragments[0]

    j,b,r = get_join_list(fragments[0])
    to_join += j
    del_atoms += r
    bonds += b
    offset = fragments[0].GetNumAtoms()

    for f in fragments[1:]:

        j,b,r = get_join_list(f)
        p = to_join.pop()
        pb = bonds.pop()

        # Check bond types if b[:-1] == pb
        if b[:-1] != pb:
            assert("Can't connect bonds")



        pairs.append((p, j[-1] + offset,pb))

        for x in j[:-1]:
            to_join.append(x + offset)
        for x in r:
            del_atoms.append(x + offset)
        bonds += b[:-1]

        offset += f.GetNumAtoms()
        new_mol = Chem.CombineMols(new_mol, f)


    new_mol =  Chem.EditableMol(new_mol)

    for a1,a2,b in pairs:
        new_mol.AddBond(a1,a2, order=b)

    # Remove atom with greatest number first:
    for s in sorted(del_atoms, reverse=True):
        new_mol.RemoveAtom(s)
    return new_mol.GetMol()





# Decide the class of a fragment
#   Either R-group, Linker or Scaffold
def get_class(fragment):

    is_ring = False
    n = 0

    for a in fragment.GetAtoms():
        if a.IsInRing():
            is_ring = True

        if a.GetAtomicNum() >= MOL_SPLIT_START:
            n += 1

    smi = Chem.MolToSmiles(fragment)

    if n == 1:
        cl = "R-group"
    elif is_ring:
        cl = "Scaffold-" + str(n)
    else:
        cl = "Linker-" + str(n)

    return cl




# Enforce conditions on fragments
def should_use(fragment):

    n = 0
    m = 0
    for a in fragment.GetAtoms():
        m += 1
        if a.GetAtomicNum() >= MOL_SPLIT_START:
            n += 1
        if n > MAX_FREE or m > MAX_ATOMS:
            return False

    return True




# Split a list of molecules into fragments.
def get_fragments(mols):

    used_mols = np.zeros(len(mols)) != 0

    fragments = dict()

    # Get all non-ring single bonds (including to H) and store in list (listofsinglebonds)
    i = -1
    for mol in mols:
        i += 1
        try:
            fs = split_molecule(mol)
        except:
            continue

        if len(fs) <= MAX_FRAGMENTS and all(map(should_use, fs)):
            used_mols[i] = True
        else:
            continue

        for f in fs:
            cl = get_class(f)
            fragments[Chem.MolToSmiles(f)] = (f, cl)

    return fragments, used_mols
    
    
import numpy as np
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import BRICS

from .conversion import mol_from_smiles, mol_to_smiles

dummy = Chem.MolFromSmiles('[*]')


def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol


def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res


def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms



def fragment_recursive(mol, frags):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))

        if bonds == []:
            frags.append(mol)
            return frags

        idxs, labs = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]

        broken = Chem.FragmentOnBonds(mol,
                                      bondIndices=[bond_idxs[0]],
                                      dummyLabels=[(0, 0)])
        head, tail = Chem.GetMolFrags(broken, asMols=True)
        print(mol_to_smiles(head), mol_to_smiles(tail))
        frags.append(head)

        fragment_recursive(tail, frags)
    except Exception:
        pass


def join_molecules(molA, molB):
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            neigh = atom.GetNeighbors()[0]
            break
    neigh = 0 if neigh is None else neigh.GetIdx()

    if marked is not None:
        ed = Chem.EditableMol(molA)
        ed.RemoveAtom(marked)
        molA = ed.GetMol()

    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh,
        useChirality=False)[0]

    Chem.Kekulize(joined)
    return joined


def has_dummy_atom(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return True
    return False


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    try:
        if count_dummies(frags[0]) != 1:
            return None, None

        if count_dummies(frags[-1]) != 1:
            return None, None

        for frag in frags[1:-1]:
            if count_dummies(frag) != 2:
                return None, None
        
        mol = join_molecules(frags[0], frags[1])
        for i, frag in enumerate(frags[2:]):
            print(i, mol_to_smiles(frag), mol_to_smiles(mol))
            mol = join_molecules(mol, frag)
            print(i, mol_to_smiles(mol))

        # see if there are kekulization/valence errors
        mol_to_smiles(mol)

        return mol, frags
    except Exception:
        return None, None
