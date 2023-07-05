import argparse
import itertools
import logging
import pathlib
from collections import abc

import atomlite
import numpy as np
import pywindow
import stk


def get_building_blocks(
    path: pathlib.Path,
    functional_group_factory: stk.FunctionalGroupFactory,
) -> abc.Iterator[stk.BuildingBlock]:
    with open(path, "r") as f:
        content = f.readlines()

    for smiles in content:
        yield stk.BuildingBlock(smiles, [functional_group_factory])


def get_initial_population(
    aldehydes: abc.Iterable[stk.BuildingBlock],
    amines: abc.Iterable[stk.BuildingBlock],
    generator: np.random.Generator,
) -> abc.Iterator[stk.MoleculeRecord]:
    for aldehyde, amine in itertools.product(aldehydes, amines):
        yield stk.MoleculeRecord(
            topology_graph=stk.cage.FourPlusSix(
                building_blocks=[aldehyde, amine],
                vertex_alignments={
                    0: generator.integers(0, 2),
                    1: generator.integers(0, 2),
                    2: generator.integers(0, 2),
                    3: generator.integers(0, 2),
                },
                optimizer=stk.MCHammer(),
            ),
        )


def get_key(molecule: stk.Molecule) -> str:
    pass


def pore_diameter(
    database: atomlite.Database,
    molecule: stk.Molecule,
) -> float:
    key = get_key(molecule)
    path = "$.gokay_ea.pore_diameter"
    pore_diamater = database.get_property(key, path)
    if pore_diamater is None:
        pw_mol = pywindow.Molecule.load_rdkit_mol(molecule.to_rdkit_mol())
        pore_diameter = pw_mol.calculate_pore_diameter()
        database.set_property(key, path, pore_diameter, commit=False)
    return pore_diameter


def is_aldehyde(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Aldehyde


def is_amine(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.PrimaryAmino


def get_num_functional_groups(building_block: stk.BuildingBlock) -> int:
    return building_block.get_num_functional_groups()


def normalize_generations(
    fitness_calculator: stk.FitnessCalculator,
    fitness_normalizer: stk.FitnessNormalizer,
    generations: abc.Sequence[stk.Generation],
) -> abc.Iterator[stk.Generation]:
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
    population_size = sum(1 for _ in generations[0].get_molecule_records())
    num_molecules = num_generations * population_size

    for generation, start in zip(
        generations,
        range(0, num_molecules, population_size),
    ):
        end = start + population_size
        yield stk.Generation(
            molecule_records=population[start:end],
            mutation_records=tuple(generation.get_mutation_records()),
            crossover_records=tuple(generation.get_crossover_records()),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "aldehydes",
        help="SMILES of tritopic aldehyde building blocks.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "amines",
        help="SMILES of ditopic amine building blocks.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "database",
        help="An AtomLite database which holds the results of the EA.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "random_seed",
        help="The random seed to use for deterministic results.",
        type=int,
        default=4,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    generator = np.random.default_rng(args.random_seed)
    logging.info("Making building blocks.")
    aldehydes = tuple(
        get_building_blocks(
            path=args.aldehydes,
            functional_group_factory=stk.AldehydeFactory(),
        )
    )
    amines = tuple(
        get_building_blocks(
            path=args.amines,
            functional_group_factory=stk.PrimaryAminoFactory(),
        )
    )
    initial_population = tuple(
        get_initial_population(aldehydes[:5], amines[:5], generator)
    )

    db = stk.ConstructedMoleculeMongoDb(client)
    fitness_db = stk.ValueMongoDb(client, "fitness_values")
    fitness_calculator = stk.FitnessFunction(
        fitness_function=pore_diameter,
    )

    generation_selector = stk.Best(
        num_batches=10,
        duplicate_molecules=False,
    )
    stk.SelectionPlotter(
        filename="generation_selection",
        selector=generation_selector,
    )

    mutation_selector = stk.Roulette(
        num_batches=4,
        generator=generator,
    )
    stk.SelectionPlotter("mutation_selection", mutation_selector)

    crossover_selector = stk.Roulette(
        num_batches=5,
        batch_size=2,
        generator=generator,
    )
    stk.SelectionPlotter("crossover_selection", crossover_selector)

    fitness_normalizer = stk.NormalizerSequence(
        fitness_normalizers=(
            stk.Add(0.1),
            stk.DivideByMean(),
            stk.Power(-1),
        ),
    )
    ea = stk.EvolutionaryAlgorithm(
        num_processes=1,
        initial_population=initial_population,
        fitness_calculator=fitness_calculator,
        mutator=stk.RandomMutator(
            mutators=(
                stk.RandomBuildingBlock(
                    building_blocks=aldehydes,
                    is_replaceable=is_aldehyde,
                    generator=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=aldehydes,
                    is_replaceable=is_aldehyde,
                    generator=generator,
                ),
                stk.RandomBuildingBlock(
                    building_blocks=amines,
                    is_replaceable=is_amine,
                    generator=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=amines,
                    is_replaceable=is_amine,
                    generator=generator,
                ),
            ),
            generator=generator,
        ),
        crosser=stk.GeneticRecombination(get_gene=get_num_functional_groups),
        generation_selector=generation_selector,
        mutation_selector=mutation_selector,
        crossover_selector=crossover_selector,
        fitness_normalizer=fitness_normalizer,
    )

    logging.info("Starting EA.")

    generations = []
    for generation in ea.get_generations(100):
        for record in generation.get_molecule_records():
            db.put(record.get_molecule())
        generations.append(generation)

    logging.info("Making fitness plot.")

    generations = tuple(
        normalize_generations(
            fitness_calculator=fitness_calculator,
            fitness_normalizer=fitness_normalizer,
            generations=generations,
        )
    )

    fitness_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record: record.get_fitness_value(),
        y_label="Fitness Value",
    )
    fitness_progress.get_plot_data().to_csv("fitness_progress.csv")
    fitness_progress.write("fitness_progress.png")

    logging.info("Making rotatable bonds plot.")

    rotatable_bonds_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record: pore_diameter(record.get_molecule()),
        y_label="Pore diameter",
    )
    rotatable_bonds_progress.write("pore_diameter_progress.png")


if __name__ == "__main__":
    main()
