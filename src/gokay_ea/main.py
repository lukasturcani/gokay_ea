import sys


from rdkit import Chem
from rdkit.Chem import AllChem as rdkit
from collections import defaultdict
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDistGeom
IPythonConsole.ipython_3d = True

import py3Dmol
from IPython.display import Image
import matplotlib.pyplot as plt
import subprocess
import time
import stk
import stko
import spindry as spd
from itertools import product
import os
import stk
import rdkit.Chem.AllChem as rdkit
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit import RDLogger
import pymongo
import numpy as np
import itertools as it
import logging
import argparse
import pathlib
from rdkit.Chem import AllChem
import pywindow
import random


######## POPULATION DEFINITION ###########
#
#
#
import warnings



def get_building_blocks(path, functional_group_factory):
    with open(path, 'r') as f:
        content = f.readlines()

    for smiles in content:

        molecule = rdkit.AddHs(rdkit.MolFromSmiles(smiles)) #noqa
        rdkit.SanitizeMol(molecule)
        #rdkit.Kekulize(molecule) #noqa
        AllChem.EmbedMolecule(molecule) #noqa
        AllChem.AssignStereochemistry(molecule) #noqa



        #isomeric smiles true kısmı eklendi ,isomericSmiles=True
        #molecule.AddConformer(
        #    conf=rdkit.Conformer(molecule.GetNumAtoms()),
        #)

        #print(molecule.GetConformer().GetPositions())

        building_block = stk.BuildingBlock.init_from_rdkit_mol(
            molecule=molecule,
            functional_groups=[functional_group_factory],
        )
        yield (building_block)
             #get position matrix çıktı

def get_initial_population(alde, amine):
    for amin_eee, ald_e in it.product(alde[:5], amine[:5]):
        #print(amin_eee)
        #print(ald_e)
        yield stk.MoleculeRecord(
            topology_graph=stk.cage.FourPlusSix(
            building_blocks={
                amin_eee: range(4),
                ald_e: range(4,10),
        },
            vertex_alignments={0: random.randint(0, 2), 1: random.randint(0, 2), 2: random.randint(0, 2), 3: random.randint(0, 2)},
            optimizer=stk.MCHammer(),
            ),
        )




#
#
######## POPULATION DEFINITION ###########



#######################################property def#######################################
#
#
#



def pore_diameter(mol):
    pw_mol = pywindow.Molecule.load_rdkit_mol(mol.to_rdkit_mol()) #noqa
    mol.pore_diameter = pw_mol.calculate_pore_diameter()
    return mol.pore_diameter

def pore_diameter2(mol):
    pw_mol = pywindow.Molecule.load_rdkit_mol(mol.to_rdkit_mol()) #noqa
    mol.pore_diameter = pw_mol.calculate_pore_diameter()
    return (mol.pore_diameter)**2



def window_std(mol):
    pw_mol = pywindow.Molecule.load_rdkit_mol(mol.to_rdkit_mol())
    windows = pw_mol.calculate_windows()
    mol.window_std = None
    if windows is not None and len(windows) > 3:
        mol.window_std = np.std(windows)
    return mol.window_std



def fingerprint(mol):
    rdkit_mol = mol.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_mol)
    info = {}
    fp = rdkit.GetMorganFingerprintAsBitVect(
        mol=rdkit_mol,
        radius=8,
        nBits=512,
        bitInfo=info,
    )
    fp = list(fp)
    for bit, activators in info.items():
        fp[bit] = len(activators)
    return np.mean([fp])
""""
def get_num_rotatable_bonds(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return rdkit.CalcNumRotatableBonds(rdkit_molecule)


def get_complexity(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return BertzCT(rdkit_molecule)


def get_num_bad_rings(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return sum(
        1
        for ring in rdkit.GetSymmSSSR(rdkit_molecule)
        if len(ring) < 5
    )
"""
#
#
#
#######################################property def#######################################





######## func group definition ###########
#
#
#
def get_functional_group_type(building_block):
    functional_group, = building_block.get_functional_groups(0) ###### virgüllerrrr
    return functional_group.__class__

def get_topology_type(building_block): #crossover kısmında kullanılabilir
    topology_group = building_block.get_topology_type(0)
    return topology_group.__class__


def is_alde(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Aldehyde


def is_amine(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.PrimaryAmino

def get_num_functional_groups(building_block):
   return building_block.get_num_functional_groups()

###  NEW ####

def get_polymorphs_4_plus_6(building_block):

    broken_bonds_by_id = []
    disconnectors = []
    for bi in cage1.get_bond_infos():
        if bi.get_building_block() is None:
            a1id = bi.get_bond().get_atom1().get_id()
            a2id = bi.get_bond().get_atom2().get_id()
            broken_bonds_by_id.append(sorted((a1id, a2id)))
            disconnectors.extend((a1id, a2id))
    new_topology_graph = stko.TopologyExtractor()
    tg_info = new_topology_graph.extract_topology(
        molecule=building_block,
        broken_bonds_by_id=broken_bonds_by_id,
        disconnectors=set(disconnectors),
    )
    return tg_info.get_connectivities()

#
#
#
######## func group definition ###########

######## normalize ###########
#
#
#
def normalize_generations(
    fitness_calculator,
    fitness_normalizer,
    generations,
):
    population = tuple(
        record.with_fitness_value(
            fitness_value=fitness_calculator.get_fitness_value(
                molecule=record.get_molecule(),
            ),
            normalized=False,
        )
        for generation in generations
        for record in generation.get_molecule_records()
    )
    population = tuple(fitness_normalizer.normalize(population))

    num_generations = len(generations)
    population_size = sum(
        1 for _ in generations[0].get_molecule_records()
    )
    num_molecules = num_generations*population_size

    for generation, start in zip(
        generations,
        range(0, num_molecules, population_size),
    ):
        end = start + population_size
        yield stk.Generation(
            molecule_records=population[start:end],
            mutation_records=tuple(
                generation.get_mutation_records()
            ),
            crossover_records=tuple(
                generation.get_crossover_records()
            ),
        )

#
#
#
######## normalize ###########


##### writer######
#
#
def write(molecule, path):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit_molecule = rdkit_molecule   #rdkit.RemoveHs(rdkit_molecule)   #SWITCHED AND TURNED OFF
    building_block = stk.BuildingBlock.init_from_rdkit_mol(
        molecule=rdkit_molecule,
    )
    building_block.write(path) #modified

#
#
##### writer######



######## MAIN #####
#
#
#
def main():
    logging.basicConfig(level=logging.INFO)

    # Use a random seed to get reproducible results.
    random_seed = 4
    #generator = np.random.RandomState(random_seed) turned off
    logging.info('Making building blocks.')

    # Load the building block databases.
    alde = tuple(get_building_blocks(
        path=pathlib.Path(__file__).parent / 'alde',
        functional_group_factory=stk.AldehydeFactory(),
    ))
    amine = tuple(get_building_blocks(
        path=pathlib.Path(__file__).parent / 'amine',
        functional_group_factory=stk.PrimaryAminoFactory(),
    ))

    initial_population = tuple(
        get_initial_population(alde, amine)
    )
    # Write the initial population.
    for i, record in enumerate(initial_population):
        write(record.get_molecule(), f'initial_{i}.mol') #mol writer modified

    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = stk.ConstructedMoleculeMongoDb(client)
    fitness_db = stk.ValueMongoDb(client, 'fitness_values')
    fitness_calculator = stk.PropertyVector(
        property_functions=(
            pore_diameter,
            window_std,
            pore_diameter2,
        ),
        input_database=fitness_db,
        output_database=fitness_db,
    )
    print(fitness_calculator)

    # Plot selections.
    generation_selector = stk.Best(
        num_batches=10,
        duplicate_molecules=False, #true or false modified false
    )
    stk.SelectionPlotter(
        filename='generation_selection',
        selector=generation_selector,
    )

    mutation_selector = stk.Roulette(
        num_batches=4,
        random_seed=21,
    )
    stk.SelectionPlotter('mutation_selection', mutation_selector)

    crossover_selector = stk.Roulette(
        num_batches=5, #modified org 5
        batch_size=2, #modified org 2
        random_seed=21,
    )
    stk.SelectionPlotter('crossover_selection', crossover_selector)



    fitness_normalizer = stk.NormalizerSequence(
        fitness_normalizers=(
            stk.Add((0.1, 0.1,0.1)),
            stk.Power((2,2,2)),
            stk.Multiply((1, 1,1)), #burayı iki dimentional yapmak düzeltebiliyor ancak baya bi error çıkıyore
            stk.DivideByMean(),
            stk.Sum(),
            stk.Power(-1),
        ),
    )
############## CROSSER AND MUTATOR  #################
##
##
##
    ea = stk.EvolutionaryAlgorithm(
        num_processes=1,
        initial_population=initial_population,
        fitness_calculator=fitness_calculator,
        mutator=stk.RandomMutator(
            mutators=(

                stk.RandomBuildingBlock(
                    building_blocks=alde,
                    is_replaceable=is_alde,
                    random_seed=21,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=alde,
                    is_replaceable=is_alde,
                    random_seed=21,
                ),
                stk.RandomBuildingBlock(
                    building_blocks=amine,
                    is_replaceable=is_amine,
                    random_seed=21,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=amine,
                    is_replaceable=is_amine,
                    random_seed=21,
                ),
            ),
            random_seed=21,
        ),
        crosser=stk.GeneticRecombination(
            #get_gene=get_functional_group_type,
            get_gene=get_num_functional_groups
            #get_gene= get_topology_type
        ),


        generation_selector=generation_selector,
        mutation_selector=mutation_selector,
        crossover_selector=crossover_selector,
        fitness_normalizer=fitness_normalizer,
    )

    logging.info('Starting EA.')

    generations = []
    for generation in ea.get_generations(100):
        for record in generation.get_molecule_records():
            db.put(record.get_molecule())
        generations.append(generation)

    # Write the final population.
    for i, record in enumerate(generation.get_molecule_records()):
        write(record.get_molecule(), f'final_{i}.mol')

    logging.info('Making fitness plot.')

    # Normalize the fitness values across the entire EA before
    # plotting the fitness values.
    generations = tuple(normalize_generations(
        fitness_calculator=fitness_calculator,
        fitness_normalizer=fitness_normalizer,
        generations=generations,
    ))

    fitness_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record: record.get_fitness_value(),
        y_label='Fitness Value',
    )
    fitness_progress.get_plot_data().to_csv('fitness_progress.csv')
    fitness_progress.write('fitness_progress.png')

    logging.info('Making rotatable bonds plot.')

    rotatable_bonds_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record:
            pore_diameter(record.get_molecule()),
        y_label='Pore diameter',
    )
    rotatable_bonds_progress.write('pore_diameter_progress.png')


if __name__ == '__main__':
    """parser = argparse.ArgumentParser(
        description="EA script gokay"
    )
    parser.add_argument("-db", help="Contains URL for MongoDB.", required=True)
    args = parser.parse_args()
    main(args)"""
    main()
############## CROSSER AND MUTATOR  #################

################   OPTIMIZER   ######################
##
##
##

"""

optimizer = stk.Sequence(
    stk.MacroModelForceField(
        macromodel_path='/home/lt912/schrodinger2017-4',
        restricted=True,
        use_cache=True,
    ),
    stk.MacroModelForceField(
        macromodel_path='/home/lt912/schrodinger2017-4',
        restricted=False,
        use_cache=True,
    ),
    use_cache=True,
)


"""
################   OPTIMIZER   ######################



def main():
    pass


if __name__ == "__main__":
    main()
