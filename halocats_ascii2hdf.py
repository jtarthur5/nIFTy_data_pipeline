# Script for converting and reading halo_catalog files

import os
import time
import numpy as np
import pandas as pd
import h5py

# --- Functions --- #


def get_cluster_dirs(sim_dir):
    """Get directory list of all clusters"""

    # Get directories
    cluster_dirs = os.listdir(sim_dir)

    # Delete relevant directories
    idxs = np.array([27, 29, 30], dtype=np.int32)
    for index in sorted(idxs, reverse=True):
        del cluster_dirs[index]
    print(cluster_dirs)

    # Get full path names
    cluster_names = cluster_dirs
    cluster_dirs = [sim_dir + s for s in cluster_dirs]

    # Get long cluster name string for attrs.
    long_cluster_name = cluster_names[0]
    for i in range(1, len(cluster_names)):
        long_cluster_name += ' ' + cluster_names[i]
    return cluster_dirs, cluster_names, long_cluster_name


def get_velociraptor_pandas(filename):
    """Load velociraptor catalog from ascii with Pandas."""

    # Open
    halofile = open(filename)
    next(halofile)
    next(halofile)
    names = ((halofile.readline())).split()
    fieldnames = [fieldname.split("(")[0] for fieldname in names]
    sizecheck = ((halofile.readline())).split()

    # Read in ASCII file into data frame.
    #usefulcols = np.arange(0, 83)
    # Check whether there is anything in the file.
    if np.size(sizecheck):
        catalog = pd.read_csv(filename, skiprows=2,
                              sep=' ')
    else:
        catalog = 0
    halofile.close()
    return catalog


def store_header(basefilename, hf):
    """Create header."""
    print('Populating header...')

    # Create header group and populate.
    header_grp = hf.create_group('Header')
    header_grp.attrs['Title'] = title
    header_grp.attrs['Description'] = descrip
    header_grp.attrs['BoxSize'] = box_size
    header_grp.attrs['Nsnap'] = n_snap

    # Create subgroups - halofind
    halo_header_grp = header_grp.create_group('HaloFinder')
    halo_header_grp.attrs['Name'] = halofind_name
    halo_header_grp.attrs['Version'] = halofind_vers
    halo_header_grp.attrs['Particle_Num_Threshold'] = part_threshold

    # treebuild
    tree_header_grp = header_grp.create_group('TreeBuilder')
    tree_header_grp.attrs['Name'] = tree_name
    tree_header_grp.attrs['Version'] = tree_vers
    tree_header_grp.attrs['Temporal_Linking_Length'] = tree_ll

    # cosmology
    cos_header_grp = header_grp.create_group('Cosmology')
    cos_header_grp.attrs['OmegaBaryon'] = omegab
    cos_header_grp.attrs['OmegaCDM'] = omegam
    cos_header_grp.attrs['OmegaLambda'] = omegal
    cos_header_grp.attrs['H100'] = h100
    cos_header_grp.attrs['Sigma8'] = sigma8

    # Units
    units_header_grp = header_grp.create_group('Units')
    units_header_grp.attrs['UnitLength_in_Mpc'] = unit_length
    units_header_grp.attrs['Unitvelocity_in_km/s'] = unit_veloc
    units_header_grp.attrs['Unitmass_in_Msol'] = unit_mass
    units_header_grp.attrs['Flag_Physical_Comoving'] = flag_comoving
    units_header_grp.attrs['Flag_Hubble_Flow_Included'] = flag_hubbleflow

    # Part types
    partypes_header_grp = header_grp.create_group('PartTypes')
    partypes_header_grp.attrs['Flag_Gas'] = flag_gas
    partypes_header_grp.attrs['Flag_Star'] = flag_star
    partypes_header_grp.attrs['Flag_BH'] = flag_bh


def store_halos(basefilename, snapnum, hf, snapnum1, snapcount, treeinfo):
    """ Store halo catalogs for each snapshot."""

    # Call file
    fname = basefilename + snapnum1 + '.VELOCIraptor.properties'
    print('Converting:', snapnum1 + '.VELOCIraptor.properties')
    halofile = open(fname, 'r')

    # Read header
    [filenum, numfiles] = halofile.readline().split()
    filenum = int(filenum)
    numfiles = int(numfiles)
    [numhalos, numtothalos] = halofile.readline().split()
    numhalos = np.uint64(numhalos)
    numtothalos = np.uint64(numtothalos)
    names = ((halofile.readline())).split()
    fieldnames = [fieldname.split("(")[0] for fieldname in names]
    fieldnames.append('-')
    fieldnames.append('TreeIndex')
    fieldnames1 = fieldnames[0]
    for i in fieldnames[1:]:
        fieldnames1 = fieldnames1 + ' ' + i
    halofile.close()

    # Read ascii file into memory
    catalog = get_velociraptor_pandas(fname)

    # Add tree indexes to halo catalog - catch errors if snaps arent in tree (IndexError)
    # or the catalogs do not have any halos in them (TypeError)
    try:
        block_indx = (np.size(treeinfo['Snapshot']) + 1) - snapcount
        start_indx = treeinfo['StartSnapBlockIndx'][block_indx]
        treeindx = catalog['ID(1)'] + (start_indx - 1)
        catalog = pd.concat([catalog, treeindx], axis=1)
    except (IndexError, TypeError):
        print('Skipping tree indexes for', snapnum1, 'no halos in catalog.')

    # Add Snapshot group.
    if snapnum1 == 'snap_000':
        hf.create_group('Snapshots')

    # Add current snap group - have to correct snapnums
    # as tree missing snap, use snapnum1.
    grp_snap = hf['Snapshots/'].create_group(snapnum1)
    grp_snap.attrs['Snapnum'] = snapnum1
    grp_snap.attrs['NHalos'] = numtothalos
    grp_snap.attrs['ScaleFactor'] = 1  # Where is this stored?
    grp_snap.attrs['FieldNames'] = fieldnames1

    # Store halo catalog
    grp_snap.create_dataset('HaloCatalog', data=catalog)


def store_tree(basefilename, hf):
    """Store tree data."""
    print('Populating Tree...')

    # Read in ascii tree file
    tree_fname = basefilename + 'VELOCIraptor.tree'
    t0 = time.time()
    treefile = open(tree_fname, 'r')
    # treefile.readline()
    tree_num_snaps = (int(treefile.readline()) - 1)
    treefile.readline()
    numtothalos = int(treefile.readline())

    # Preallocate tree dict and treeinfo, which will is used to speed up
    # storing treeidxs in halo catalogs.
    tree = {'treeID': np.ones(numtothalos, dtype=np.int64) * -1,
            "HaloSnapID": np.ones(numtothalos, dtype=np.int64) * -1,
            "HaloSnapNum": np.ones(numtothalos, dtype=np.int32) * -1,
            "HaloSnapIndex": np.ones(numtothalos, dtype=np.int64) * -1,
            'ProgenTreeID': np.ones(numtothalos, dtype=np.int64) * -1,
            'ProgenSnapID': np.ones(numtothalos, dtype=np.int64) * -1,
            'ProgenSnapNum': np.ones(numtothalos, dtype=np.int32) * -1,
            'ProgenIndex': np.ones(numtothalos, dtype=np.int64) * -1,
            'DescTreeID': np.ones(numtothalos, dtype=np.int64) * -1,
            'DescSnapID': np.ones(numtothalos, dtype=np.int64) * -1,
            'DescSnapNum': np.ones(numtothalos, dtype=np.int32) * -1,
            'DescIndex': np.ones(numtothalos, dtype=np.int64) * -1,
            'RootProgenTreeID': np.ones(numtothalos, dtype=np.int64) * -1,
            'RootProgenSnapID': np.ones(numtothalos, dtype=np.int64) * -1,
            'RootProgenSnapNum': np.ones(numtothalos, dtype=np.int32) * -1,
            'RootProgenIndex': np.ones(numtothalos, dtype=np.int64) * -1,
            'RootDescTreeID': np.ones(numtothalos, dtype=np.int64) * -1,
            'RootDescSnapID': np.ones(numtothalos, dtype=np.int64) * -1,
            'RootDescSnapNum': np.ones(numtothalos, dtype=np.int32) * -1,
            'RootDescIndex': np.ones(numtothalos, dtype=np.int64) * -1,
            'NumProgen': np.ones(numtothalos, dtype=np.int32) * -1}
    treeinfo = {'Snapshot': np.zeros(tree_num_snaps - 1, dtype=np.int32),
                'NumHalos': np.zeros(tree_num_snaps - 1, dtype=np.int32),
                'StartSnapBlockIndx': np.zeros(tree_num_snaps - 1, dtype=np.int64)}

    # Read tree and process main progenitor.
    loop1 = 0
    loop2 = 0
    for i in range(tree_num_snaps - 1):
        [snapval, numhalos] = treefile.readline().strip().split('\t')
        snapval = int(snapval)
        numhalos = int(numhalos)

        # Store tree info for later us.
        treeinfo['Snapshot'][loop1] = snapval
        treeinfo['StartSnapBlockIndx'][loop1] = np.sum(
            treeinfo['NumHalos'][:loop1])
        treeinfo['NumHalos'][loop1] = numhalos
        loop1 += 1

        # Store main progen for each halo in snap.
        for j in range(numhalos):
            [hid, nprog] = treefile.readline().strip().split('\t')
            hid = np.int64(hid)
            nprog = int(nprog)
            tree['treeID'][loop2] = hid
            tree['HaloSnapNum'][loop2] = snapval
            tree['NumProgen'][loop2] = nprog
            if (nprog > 0):
                for k in range(nprog):
                    progen_id = np.int64(treefile.readline())
                    # print(progen_id)
                    if k == 0:
                        tree['ProgenTreeID'][loop2] = progen_id
                        tree['ProgenSnapNum'][loop2] = int(
                            str(progen_id)[:-10])
                        # print(int(str(progen_id)[:-10]))
            loop2 += 1
    print('Time taken process progenitors:',
          time.time() - t0, 's')
    treefile.close()

    # Read tree file again and process descendants.
    t0 = time.time()
    treefile = open(tree_fname, 'r')
    for i in range(3):
        treefile.readline()
    for i in range(tree_num_snaps - 1):
        [snapval, numhalos] = treefile.readline().strip().split('\t')
        snapval = int(snapval)
        numhalos = int(numhalos)
        for j in range(numhalos):
            [hid, nprog] = treefile.readline().strip().split('\t')
            hid = np.int64(hid)
            nprog = int(nprog)
            if (nprog > 0):
                for k in range(nprog):
                    progen_id = np.int64(treefile.readline())
                    progen_snapid = progen_id % (
                        np.power(10, (np.int64(np.floor(np.log10(progen_id))) - 2)))
                    progen_snapnum = int(str(progen_id)[:-10])
                    block_indx = (
                        np.size(treeinfo['Snapshot']) + 1) - progen_snapnum
                    start_indx = treeinfo['StartSnapBlockIndx'][block_indx]
                    tree['DescTreeID'][start_indx + (progen_snapid - 1)] = hid
                    tree['DescSnapNum'][start_indx +
                                        (progen_snapid - 1)] = snapval
    print('Time taken to process descendants:',
          time.time() - t0, 's')
    treefile.close()

    # Find snap ids.
    tree['HaloSnapID'] = tree['treeID'] % (
        np.power(10, (np.int64(np.floor(np.log10(tree['treeID']))) - 2)))
    tree['ProgenSnapID'] = tree['ProgenTreeID'] % (
        np.power(10, (np.int64(np.floor(np.log10(tree['ProgenTreeID']))) - 2)))
    tree['DescSnapID'] = tree['DescTreeID'] % (
        np.power(10, (np.int64(np.floor(np.log10(tree['DescTreeID']))) - 2)))

    # Find progenitor and descendant indexes.
    t0 = time.time()
    for i in range(0, np.size(tree['treeID'])):
        # Find indexes of progenitors.
        progen_id = tree['ProgenTreeID'][i]
        if progen_id != -1:
            progen_snapid = tree['ProgenSnapID'][i]
            progen_snapnum = tree['ProgenSnapNum'][i]
            block_indx = (
                np.size(treeinfo['Snapshot']) + 1) - progen_snapnum
            start_indx = treeinfo['StartSnapBlockIndx'][block_indx]
            tree['ProgenIndex'][i] = start_indx + (progen_snapid - 1)
        # Find indexes of descendants.
        desc_id = tree['DescTreeID'][i]
        if desc_id != -1:
            desc_snapid = tree['DescSnapID'][i]
            desc_snapnum = tree['DescSnapNum'][i]
            block_indx = (
                np.size(treeinfo['Snapshot']) + 1) - desc_snapnum
            start_indx = treeinfo['StartSnapBlockIndx'][block_indx]
            tree['DescIndex'][i] = start_indx + (desc_snapid - 1)
    print('Time taken to catalog progen and desc indexes:', time.time() - t0, 's')

    # Catalog roots - descendants first.
    t0 = time.time()
    for i in range(0, np.size(tree['treeID'])):
        j = i
        desc_id = tree['DescTreeID'][i]
        if desc_id != -1:
            while desc_id != -1:
                k = j
                desc_id = tree['DescTreeID'][j]
                j = tree['DescIndex'][j]
            tree['RootDescTreeID'][i] = tree['treeID'][k]
            tree['RootDescSnapNum'][i] = int(
                str(tree['RootDescTreeID'][i])[:-10])
            tree['RootDescIndex'][i] = k
    print('Time taken to catalog root descendants:', time.time() - t0, 's')

    # Catalog roots - main progenitor root.
    t0 = time.time()
    for i in range(0, np.size(tree['treeID'])):
        j = i
        progen_id = tree['ProgenTreeID'][i]
        if progen_id != -1:
            while progen_id != -1:
                k = j
                progen_id = tree['ProgenTreeID'][j]
                j = tree['ProgenIndex'][j]
            tree['RootProgenTreeID'][i] = tree['treeID'][k]
            tree['RootProgenSnapNum'][i] = int(
                str(tree['RootProgenTreeID'][i])[:-10])
            tree['RootProgenIndex'][i] = k
    print('Time taken to catalog root progenitors:', time.time() - t0, 's')

    tree['RootDescSnapID'] = tree['RootDescTreeID'] % (
        np.power(10, (np.int64(np.floor(np.log10(tree['RootDescTreeID']))) - 2)))
    tree['RootProgenSnapID'] = tree['RootProgenTreeID'] % (
        np.power(10, (np.int64(np.floor(np.log10(tree['RootProgenTreeID']))) - 2)))

    # Printing tail end of tree, useful with ascii .tree for checks.
    print(np.size(tree['treeID']))
    print(tree['treeID'][-20:])
    print(tree['HaloSnapID'][-20:])
    print(tree['HaloSnapNum'][-20:])
    print('\n')
    print(tree['ProgenTreeID'][-20:])
    print(tree['ProgenSnapID'][-20:])
    print(tree['ProgenSnapNum'][-20:])
    print(tree['ProgenIndex'][-20:])
    print('\n')
    print(tree['DescTreeID'][-20:])
    print(tree['DescSnapID'][-20:])
    print(tree['DescSnapNum'][-20:])
    print(tree['DescIndex'][-20:])
    print('\n')
    print(tree['RootDescTreeID'][-20:])
    print(tree['RootDescSnapID'][-20:])
    print(tree['RootDescSnapNum'][-20:])
    print(tree['RootDescIndex'][-20:])
    print('\n')
    print(tree['RootProgenTreeID'][-20:])
    print(tree['RootProgenSnapID'][-20:])
    print(tree['RootProgenSnapNum'][-20:])
    print(tree['RootProgenIndex'][-20:])

    # Create tree group and store information.
    tree_grp = hf.create_group('Tree')
    tree_grp.create_dataset('TreeID', data=tree['treeID'])
    tree_grp.create_dataset('HaloSnapID', data=tree['HaloSnapID'])
    tree_grp.create_dataset('HaloSnapNum', data=tree['HaloSnapNum'])
    tree_grp.create_dataset('ProgenTreeID', data=tree['ProgenTreeID'])
    tree_grp.create_dataset('ProgenSnapID', data=tree['ProgenSnapID'])
    tree_grp.create_dataset('ProgenSnapNum', data=tree['ProgenSnapNum'])
    tree_grp.create_dataset('ProgenIndex', data=tree['ProgenIndex'])
    tree_grp.create_dataset('DescTreeID', data=tree['DescTreeID'])
    tree_grp.create_dataset('DescSnapID', data=tree['DescSnapID'])
    tree_grp.create_dataset('DescSnapNum', data=tree['DescSnapNum'])
    tree_grp.create_dataset('DescIndex', data=tree['DescIndex'])
    tree_grp.create_dataset('RootDescTreeID', data=tree['RootDescTreeID'])
    tree_grp.create_dataset('RootDescSnapID', data=tree['RootDescSnapID'])
    tree_grp.create_dataset('RootDescSnapNum', data=tree['RootDescSnapNum'])
    tree_grp.create_dataset('RootDescIndex', data=tree['RootDescIndex'])
    tree_grp.create_dataset('RootProgenTreeID', data=tree['RootProgenTreeID'])
    tree_grp.create_dataset('RootProgenSnapID', data=tree['RootProgenSnapID'])
    tree_grp.create_dataset('RootProgenSnapNum', data=tree[
                            'RootProgenSnapNum'])
    tree_grp.create_dataset('RootProgenIndex', data=tree['RootProgenIndex'])

    return treeinfo


def store_particles(basefilename, snapnum, hf):
    """ Store halo catalogs for each snapshot."""

    # Open relvant files.
    print('Converting:', snapnum)
    cat_group = basefilename + snapnum + '.VELOCIraptor.catalog_groups'
    cat_partID = basefilename + snapnum + '.VELOCIraptor.catalog_particles'
    cat_ptype = basefilename + snapnum + '.VELOCIraptor.catalog_parttypes'
    groupfile = open(cat_group, 'r')

    # Read header of one of them.
    [filenum, numfiles] = groupfile.readline().split()
    filenum = int(filenum)
    numfiles = int(numfiles)
    [numhalos, numtothalos] = groupfile.readline().split()
    numhalos = np.uint64(numhalos)
    numtothalos = np.uint64(numtothalos)
    groupfile.close()

    # Create particle group
    if snapnum == 'snap_000':
        hf.create_group('Particles')

    # Add current particle snap group.
    grp_snap = hf['Particles/'].create_group(snapnum)
    grp_snap.attrs['Snapnum'] = snapnum
    grp_snap.attrs['NHalos'] = numtothalos

    # Read in data. Safe guard against empty files.
    try:
        group_data = pd.read_csv(cat_group, dtype=np.int64,
                                 header=None, skiprows=2, delimiter='\n')
        partID_data = pd.read_csv(cat_partID, dtype=np.int64,
                                  header=None, skiprows=2, delimiter='\n')
        ptype_data = pd.read_csv(cat_ptype, dtype=np.int64,
                                 header=None, skiprows=2, delimiter='\n')

        # Store particle catalog
        grp_snap.create_dataset('GroupInfo', data=group_data)
        grp_snap.create_dataset('PartID', data=partID_data)
        grp_snap.create_dataset('PType', data=ptype_data)

    except ValueError:
        print('No particle data')

        # Store empty arrays
        grp_snap.create_dataset('GroupInfo', data=np.array([]))
        grp_snap.create_dataset('PartID', data=np.array([]))
        grp_snap.create_dataset('PType', data=np.array([]))


def main_velociraptor_convert(hdf_path, base_catalog_dir):
    """ Main converter function. """

    # Get all cluster directories.
    cluster_dirs, cluster_names, long_cluster_name = get_cluster_dirs(
        base_catalog_dir)

    # Store all clusters.
    for d in range(0, len(cluster_names)):

        # Get cluster names and directories.
        cluster = cluster_names[d]
        catalog_dir = cluster_dirs[d] + '/'
        print('Storing:', cluster)

        # Create hdf5 properties file.
        hdf_fname = hdf_path + cluster + '.hdf5'
        hf = h5py.File(hdf_fname, 'w')

        # Populate header
        store_header(catalog_dir, hf)

        # Add merger tree information to hdf5 file.
        treeinfo = store_tree(catalog_dir, hf)

        # convert properties files - loop through all snaps.
        print('Populating Halos...')
        t0 = time.time()
        for snapnum in range(0, num_snaps):
            snapcount = snapnum
            snapnum1 = 'snap_%03d' % snapnum
            store_halos(catalog_dir, snapnum, hf,
                        snapnum1, snapcount, treeinfo)
        print('Finished populating catalogs in: ', time.time() - t0, 's')

        """
        # convert particle files - loop through all snaps.
        print('Populating Particles...')
        t0 = time.time()
        for snapnum in range(0, num_snaps):
            snapnum1 = 'snap_%03d' % snapnum
            store_particles(catalog_dir, snapnum1, hf)
        print('Finished populating particles in: ', time.time() - t0, 's')
        """

        # print contents of hdf5 filename
        hf.visit(printname)
        hf.close()


def printname(name):
    """ Printing function to be used in tandem with .visit method"""
    print(name)

# --- MAIN --- #

# Header info - any VELOCIraptor output file to read this stuff info from?
title = ''
descrip = ''
box_size = '100Mpc'
n_snap = int(129)
halofind_name = 'VELOCIraptor'
halofind_vers = np.float(1.13)
part_threshold = int(20)
tree_name = 'VELOCIraptor'
tree_vers = np.float(1.13)
tree_ll = int(4)
omegab = np.float(0.0482)
omegam = np.float(0.27)
omegal = np.float(0.73)
sigma8 = np.float(0.82)
h100 = int(70)
unit_length = float(0.001)
unit_veloc = int(1)
unit_mass = float(1e10)
flag_comoving = int(0)
flag_hubbleflow = int(0)
flag_gas = int(1)
flag_star = int(1)
flag_bh = int(1)
num_snaps = int(129)

# Path to data, and where to save HDF5 file. CHANGE THIS.
hdf_path = ''
base_catalog_dir = ''

# Start conversion
start = time.time()
main_velociraptor_convert(hdf_path, base_catalog_dir)
print('Time taken to store hdf5:', time.time() - start, 's')
