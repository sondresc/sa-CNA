#!/usr/bin/python3

##############################################################
###Import packages from python
##############################################################
import os
import sys
import numpy as np
from ovito.io import import_file
import ovito.data as data
from ovito.modifiers import CommonNeighborAnalysisModifier #, CreateBondsModifier, ComputePropertyModifier

##############################################################
###Define some functions required to calculate cna
##############################################################
def surfaceatom( i, nn_list, max_nn=10, struc_ID='fcc', tolerance=0.25 ):
    r,nn_list_cut = adaptive_r( max_nn, nn_list, struc_ID ) # Find adaptive r
    cut = tolerance*r # Get the cutoff for how much above the neighbours can be and the central still count as surface
    pos_i = particle_pos[i] # Position of central 1
    nn_above = [1 for j in nn_list_cut if (particle_pos[j[0]][2]-pos_i[2]) > cut] # Check if near neighbour j is above central i
    if len(nn_above) > 0:
        return False
    else:
        return True

def distance( i,j ): # Function that calculates the distance between particle i and j
    pos_i = particle_pos[i] # Coordinates of particle i
    pos_j = particle_pos[j] # Coordinates of particle i
    vec = pos_i - pos_j # The vector between particle 
    cellx = np.linalg.norm(datainfo.cell_[:,0]) # Cell x-size 
    celly = np.linalg.norm(datainfo.cell_[:,1]) # Cell y-size 
    if abs(vec[0]) > cellx/2: # Adjust for periodic boundary in x-dierction
        if pos_j[0] > cellx/2:
            jx = - cellx + pos_j[0]
            vec[0] = pos_i[0] - jx
        elif pos_i[0] > cellx/2:
            jx = - cellx + pos_i[0]
            vec[0] = jx -  pos_j[0]
    if abs(vec[1]) > celly/2: # Adjust for periodic boundary in x-dierction
        if pos_j[1] > celly/2:
            jy = - celly + pos_j[1]
            vec[1] = pos_i[1] - jy
        elif pos_i[1] > celly/2:
            jy = - celly + pos_i[1]
            vec[1] = jy -  pos_j[1]
    return np.linalg.norm(vec)

def adaptive_r( max_nn,nn_list,struc_ID='fcc'): # Function for adaptive cutoff radius; Input max # of neighbours, neighbour list with index and distance from central particle i, structure ID (default fcc); Output: cutoff radius and nearest neighbour list (with index and distance to central)
    nn_list_cut = nn_list[:max_nn]
    pre = (1+np.sqrt(2))/2
    if struc_ID == 'fcc':
        r = pre * np.sum([ind[1] for ind in nn_list_cut])/max_nn
    elif struc_ID == 'bcc':
        r = pre * ( 2/np.sqrt(3) * np.sum([ind[1] for ind in nn_list_cut[:4]])/4 + np.sum([ind[1] for ind in nn_list_cut[4:]])/5 )

    nn_list_cut = [ind for ind in nn_list if ind[1] < r]
    #print([ind[0] for ind in nn_list_cut],r)
    return r,nn_list_cut

def find_cna( type_cn,type_nn,nn_i,r,index_nn,nnn_obj,searchCrystal  ): # Function to find cna
    '''
    type_cn :: type of central particle (int) 
    type_nn :: type of neighbour particle (int)
    nn_i :: list of nearest neighbours with indices and distances to central particle i (2D list)
    r :: cutoff radius (float)
    index_nn :: index of neighbour (int)
    nnn_obj :: neighbours of neighbour object (object)
    searchCrystal :: crystal structure we are trying to match (str :: options 'f01f01', 'f11f11', 'f01f11'
    '''
    ## Set up cutoff amount of nearest neighbours for structure search f01f01,f11f11,f01f11
    MaxCommonDict = {'f01f01' : {'struc_ID' : 'fcc','MaxNN' : 8, 'MaxCommon_AA' : 2, 'MaxCommon_AB' : 4},'f11f11' : {'struc_ID' : 'fcc','MaxNN' : 9, 'MaxCommon_AA' : 3, 'MaxCommon_AB' : 4},'f01f11' : {'struc_ID' : 'fcc','MaxNN' : 10, 'MaxCommon_AA' : 4, 'MaxCommon_AB' : 4}, 'b01b01' : {'struc_ID' : 'bcc', 'MaxNN' : 9, 'MaxCommon_AA' : 2, 'MaxCommon_AB' : 5}}

    cna_j = [0,0,0,0,type_cn,type_nn] # Cna of j; t : particle type of central particle

    nn_i_index = [ind[0] for ind in nn_i] # List of neighbours to central particle i with only indexes
    nnn_j = [nnn.index for nnn in nnn_obj if nnn.distance < r] # Indices of the nn of the nn
    common_nn = [common for common in nnn_j if common in nn_i_index] # Intersection between nn list of particle oi and its nn
    common_nn_dist = []
    for ci in common_nn:
        dist = distance( index_nn,ci )
        common_nn_dist.append([ci,dist])
    if type_cn == type_nn:
        MaxCommon = MaxCommonDict[searchCrystal]['MaxCommon_AA'] # Max nr of common neighbours if near neighbour is in the surface
    else:
        MaxCommon = MaxCommonDict[searchCrystal]['MaxCommon_AB'] # Max nr of common neighbours if near neighbour is in layer under surface
    if len(common_nn_dist) > MaxCommon:
        r,common_nn_dist = adaptive_r( MaxCommon,common_nn_dist,struc_ID=MaxCommonDict[searchCrystal]['struc_ID'] )
        #print('Common neighbours to %i after adjust cutoff %.4f: ' % (index_nn,r), common_nn_dist)
    #print('Common neighbours to %i with cutoff %.4f: ' % (index_nn,r),[ind[0] for ind in common_nn_dist])
    cna_j[0] = len(common_nn_dist) # Nr of common neighbours of i and j 
    bond_common_nn = []
    nbonds = [[0] for e in range(len(common_nn_dist))] 
    for c1,cnn in enumerate(common_nn_dist[:-1]): # Loop through the common nn list
        c2 = c1 + 1
        for cnn2 in common_nn_dist[c1+1:]:
            dist = distance( cnn[0],cnn2[0] ) # The distance between partice cnn and cnn2
            if dist < r:
                bond_common_nn.append([cnn[0],cnn2[0]]) 
                nbonds[c1][0]+=1
                nbonds[c2][0]+=1
                #print('Bond between common %i and %i with r=%.4f and dist=%.4f -> nbonds: ' % (cnn[0],cnn2[0],r,dist),nbonds) 
            c2 += 1
    cna_j[1] = len(bond_common_nn)

    maxbonds = 0
    minbonds = MaxCommon
    for c1 in range(0,len(common_nn_dist)):
        maxbonds = max(nbonds[c1][0],maxbonds)
        minbonds = min(nbonds[c1][0],minbonds)
    cna_j[2] = maxbonds
    cna_j[3] = minbonds
    return cna_j

def recognize_structure( cna_i,searchCrystal ): # Input cna for particle i and see if it fits a structure
    n = 0 # Nr of structure matches
    n_vac = 3 # Nr of allowed vacancies to still be included in structure
    if searchCrystal == 'f01f01':
        l_ideal = 8 # Ideal number of nearest neighbours for f01f01 (length of cna_i)
        cna_ideal = [[2,1,1,1,2,2],[4,2,1,1,2,1]] # cnas that's included in f01f01
    elif searchCrystal == 'f11f11':
        l_ideal = 9 # Ideal number of nearest neighbours for f11f11 (length of cna_i)
        cna_ideal = [[3,1,1,0,2,2],[4,2,1,1,2,1]] # cnas that's included in f11f11
    elif searchCrystal == 'f01f11':
        l_ideal = 10 # Ideal number of nearest neighbours for f01f11 (length of cna_i)
        cna_ideal = [[4,3,2,1,2,2],[3,1,1,0,2,2],[4,3,2,1,2,1],[4,2,2,0,2,1],[3,0,0,0,2,1]] # cnas that's included in f01f11
    elif searchCrystal == 'b01b01':
        l_ideal = 9 # Ideal number of nearest neighbours for b01f01 (length of cna_i)
        cna_ideal = [[2,1,1,1,2,2],[5,4,2,1,2,1],[4,4,2,2,2,1]] # cnas that's included in b01b01
    else:
        print('searchCrystal key in recognize_structure() invalid. Aborting')
        return 1

    l_cna = len(cna_i)
    for cna in cna_i: # Loop through cna_i and find structure matches
        if l_cna == l_ideal and cna in cna_ideal:
            n += 1 # Add 1 to nr of structure matches if cna_i has ideal length and cna_j match one of the criteria in cna_ideal
        elif ( l_cna > (l_ideal-n_vac-1) and l_cna < l_ideal ) and ( cna in cna_ideal ):
            n += 0.1 # Add 0.1 to nr of structure matches if cna_i has vacancies, but cna_j match one of the creiteria in cna_ideal

    ## See if the structure matched
    if n > (l_ideal-n_vac-1):
        structure = searchCrystal # Structure match when length of cna_i equal l_ideal and nr of matching cna_j match l_ideal within error of allowed vacancies
    elif isinstance(n,float) and round(n,1) == round((l_ideal-n_vac-1)/10,1):
        structure = '%s vacant' % searchCrystal # Vacant structure match when length of cna_i is smaller than l_ideal, but all cna_j match cna_ideal
    else:
        structure = 'other'

    return structure



##############################################################
### Begin data handling
##############################################################
infilename = sys.argv[1]
try: 
    write_newStruc = sys.argv[2]
except IndexError: 
    write_newStruc = False
datafile = import_file(infilename) #import_file('movie_ljf01f01_nvt.xyz') # Import datafile
datafile.source.data.cell_.pbc = (True,True,False) # Set boundary conditions (p p p)
n_frames = datafile.source.num_frames # Nr of frames in datafile

cna = CommonNeighborAnalysisModifier(mode = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff) # Import CNA modifier with adaptive cutoff

datafile.modifiers.append(cna) # Add the CNA modifier to the datafile

# Set up cutoff distances
cut = 1.0 # Distance limit for how high above central particle the neigbours can be 
MaxCommon = 8 # Max amount of common neighbours
 
## Loop through the frames
##############################################################
print('fcc(001)\tfcc(111)\tfcc(001)fcc(111)\tbcc(001)\tother')
newXYZ = 'newStructureID.xyz' # New xyz-file with structure types 
try: # Remove existing "new" structure files
    os.remove(newXYZ)
except OSError:
    pass
nStruc_perFrame = {'fcc(001)':[],'fcc(111)':[],'fcc(001)fcc(111)':[],'fcc(111) vacant':[],'bcc(001)':[],'other':[]}
for frame in range(n_frames):
    datainfo = datafile.compute(frame) # Access the data in the datafile
    particle_pos = datainfo.particle_properties.positions.array # Get particle positions
    particle_stype = datainfo.particle_properties.structure_type.array # Get the structure types (OTHER: 0, FCC: 1, HCP: 2 BCC: 3, ICO: 4)
    particle_type = datainfo.particle_properties.particle_type.array # Get the particle types (Cu: 1, Li:2)

    # Nearest neighbour finder
    N = 20 # Nr of nearest neighbours to find
    nnFind = data.NearestNeighborFinder(N,datainfo) # Initialize the nnFinder object for the 12 nn

    ## Find nr of particle types
    nType = [1]
    for t in particle_type:
        if t not in nType:
            nType.append(t)
    if len(nType) == 1: # Only one particle type; distinguish between bulk and surface on types
        particle_type = []
        for i in range(len(particle_pos)):
            nn_obj = nnFind.find(i) # List of nn of particle i
            nn_list = [[nn.index,nn.distance] for nn in nn_obj] # if nn.distance < r_fcc] # List of nn indices closer than fcc cutof
            surf = surfaceatom( i,nn_list ) # Check that particle i is at the surface; if nt, continue to next
            if surf != True:
                particle_type.append(1) # Assign particle type 1 for bulk
            else:
                particle_type.append(2) # Assign particle type 2 for surface
    elif len(nType) > 2:
        sys.exit("analyze_a-cna only supports up to 2 particle types")

    particle_stype_surf = {}
    N1 = 0
    # Loop over the Li particles
    for i,t in enumerate(particle_type):
        if t == 1:
            N1 += 1
        if t == 2 and particle_stype[i] == 0: # Only Li particles with arbitrary structure type
            nn_obj = nnFind.find(i) # List of nn of particle i
            nn_list = [[nn.index,nn.distance] for nn in nn_obj] # if nn.distance < r_fcc] # List of nn indices closer than fcc cutoff
            if len(nType) > 1: # Check for surface atoms if original structure has 2 particle types
                surf = surfaceatom( i,nn_list ) # Check that particle i is at the surface; if nt, continue to next
                if surf != True:
                    continue
            searchCrystalVec = [['f01f01',8],['f11f11',9],['f01f11',10],['b01b01',9]] # Order for search of crystal structures
            match_i = 0
            matched = False
            while not matched and match_i < len(searchCrystalVec): # Continue to try different structures to see if there's a match
                searchCrystal = searchCrystalVec[match_i][0] # Which crystal structure are we searching for
                if searchCrystal in ['f01f01','f11f11','f01f11']:
                    struc_ID = 'fcc' # Which overall crystal system does the searchCrystal belong to (required for adaptive_r())
                elif searchCrystal in ['b01b01']:
                    strucID = 'bcc'
                max_common = searchCrystalVec[match_i][1] # Max common nearest neighbours for searchCrystal
                r,nn_list_cut = adaptive_r( max_common,nn_list,struc_ID=struc_ID )
    
                cna_i =[] # Cna for particle i
                for nn in nn_list_cut: # Loop hrough the list of nn and find the cna triplet
                    type_nn = particle_type[nn[0]]
                    nnn_obj = nnFind.find(nn[0])
                    cna_j = find_cna(  t,type_nn,nn_list_cut,r,nn[0],nnn_obj,searchCrystal )
                    cna_i.append(cna_j)

                structure = recognize_structure( cna_i,searchCrystal )
                if structure == 1:
                    sys.exit()

                if searchCrystal in structure:
                    matched = True
                match_i += 1
            pID = 'particle%i' % i
            particle_stype_surf[pID] = {'ID':i,'cna':cna_i,'stype':structure}
            #if ( i == 2076 or i == 2086 ):
            #    print(particle_stype_surf[pID])
    
        elif t == 2:
            cna_i = []
            if particle_stype[i] == 1:
                structure = 'FCC'
            elif particle_stype[i] == 2:
                structure = 'HCP'
            elif particle_stype[i] == 3:
                structure = 'BCC'
            elif particle_stype[i] == 4:
                structure = 'ICO'
            pID = 'particle%i' % i
            particle_stype_surf[pID] = {'ID':i,'cna':cna_i,'stype':structure}
    
    n_fcc01 = 0 
    n_fcc11 = 0 
    n_fcc01fcc11 = 0
    n_bcc01 = 0
    n_vac = 0
    n_other = 0
    ls = len(particle_stype_surf)
    if ls > 0.01:
        for par in particle_stype_surf:
            if 'f01f01' in particle_stype_surf[par]['stype']:
                #ID = particle_stype_surf[par]['ID']-N1+1
                #print('%i, Li%i: ' % (particle_stype_surf[par]['ID'],ID), particle_stype_surf[par]['stype'],', ',particle_stype_surf[par]['cna'])
                n_fcc01 += 1
            elif 'f11f11' in particle_stype_surf[par]['stype']:
                #ID = particle_stype_surf[par]['ID']-N1+1
                #print('%i, Li%i: ' % (particle_stype_surf[par]['ID'],ID), particle_stype_surf[par]['stype'],', ',particle_stype_surf[par]['cna'])
                if 'vac' in particle_stype_surf[par]['stype']:
                    n_vac += 1
                n_fcc11 += 1
            elif 'f01f11' in particle_stype_surf[par]['stype']:
                #ID = particle_stype_surf[par]['ID']-N1+1
                #print('%i, Li%i: ' % (particle_stype_surf[par]['ID'],ID), particle_stype_surf[par]['stype'],', ',particle_stype_surf[par]['cna'])
                if 'vac' in particle_stype_surf[par]['stype']:
                    n_vac += 1
                n_fcc01fcc11 += 1
            elif 'b01b01' in particle_stype_surf[par]['stype']:
                n_bcc01 += 1
            elif 'other' in particle_stype_surf[par]['stype']:
                #ID = particle_stype_surf[par]['ID']-N1+1
                #print('%i, Li%i: ' % (particle_stype_surf[par]['ID'],ID), particle_stype_surf[par]['stype'],', ',particle_stype_surf[par]['cna'])
                n_other += 1
    else:
        ls = 1
    nStruc_perFrame['fcc(001)'].append(n_fcc01/ls)
    nStruc_perFrame['fcc(111)'].append(n_fcc11/ls)
    nStruc_perFrame['fcc(001)fcc(111)'].append(n_fcc01fcc11/ls)
    nStruc_perFrame['fcc(111) vacant'].append(n_vac/ls)
    nStruc_perFrame['bcc(001)'].append(n_bcc01/ls)
    nStruc_perFrame['other'].append(n_other/ls)

    print('%8.3f \t%8.3f \t%16.3f \t%8.3f \t%.3f' % (n_fcc01/ls,n_fcc11/ls,n_fcc01fcc11/ls,n_bcc01/ls,n_other/ls))

    ## Write coordinates and new sutructure identification to .xyz file
    if write_newStruc == 'True' or write_newStruc == 'true':
        with open(newXYZ,'a') as f: # Open the new structure file and append structure IDs and correpsonding coordinates
            f.write('%i\n' % len(particle_stype)) # Nr of particles
            f.write('Atoms. Timestep: %i\n' % frame) # Frame
            for i,st in enumerate(particle_stype):
                pID = 'particle%i' % i
                if pID in particle_stype_surf:
                    wStype = particle_stype_surf[pID]['stype']
                    if 'f01f01' in wStype:
                        wStype = 6
                    elif 'f11f11' in wStype:
                        wStype = 7
                    elif 'f01f11' in wStype:
                        wStype = 8
                    elif 'b01b01' in wStype:
                        wStype = 9
                    elif 'FCC' in wStype:
                        wStype = 1
                    elif 'HCP' in wStype:
                        wStype = 2
                    elif 'BCC' in wStype:
                        wStype = 3
                    elif 'ICO' in wStype:
                        wStype = 4
                    elif 'OTHER' in wStype:
                        wStype = 0
                    elif 'other' in wStype:
                        wStype = 10
                else:
                    wStype = st
                    #print('pID: %s, structure type: %i' % (pID,st))
                f.write('%i ' % wStype) # Write the structure type (from Ovito or assigned here) in 1st col in newXYZ
                f.write('%6f %6f %6f\n' % (particle_pos[i][0],particle_pos[i][1],particle_pos[i][2])) 
sum_fcc01 = np.sum(nStruc_perFrame['fcc(001)'])/n_frames
sum_fcc11 = np.sum(nStruc_perFrame['fcc(111)'])/n_frames
sum_f01f11 = np.sum(nStruc_perFrame['fcc(001)fcc(111)'])/n_frames
sum_nVac = np.sum(nStruc_perFrame['fcc(111) vacant'])/n_frames
sum_bcc01 = np.sum(nStruc_perFrame['bcc(001)'])/n_frames
sum_other = np.sum(nStruc_perFrame['other'])/n_frames

print('Avg ratio: %.3f, %.3f, %.3f, %.3f, %.3f per frame' % (sum_fcc01,sum_fcc11,sum_f01f11,sum_bcc01,sum_other))
