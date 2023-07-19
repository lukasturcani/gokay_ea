import argparse
import itertools
import logging
import pathlib
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, TypeAlias, cast

import atomlite
import numpy as np
import pywindow
import stk
import stko

MoleculeRecord: TypeAlias = stk.MoleculeRecord[stk.cage.Cage]


def get_building_blocks(
    path: pathlib.Path,
    functional_group_factory: stk.FunctionalGroupFactory,
) -> Iterator[stk.BuildingBlock]:
    with open(path, "r") as f:
        content = f.readlines()
    for smiles in content:
        yield stk.BuildingBlock(smiles, functional_group_factory)


def get_initial_population(
    aldehydes: Iterable[stk.BuildingBlock],
    amines: Iterable[stk.BuildingBlock],
    generator: np.random.Generator,
) -> Iterator[MoleculeRecord]:
    for aldehyde, amine in itertools.product(aldehydes, amines):
        yield stk.MoleculeRecord(
            topology_graph=stk.cage.FourPlusSix(
                building_blocks={
                    aldehyde: range(4),
                    amine: range(4, 10),
                },
                vertex_alignments={
                    0: generator.integers(0, 3),
                    1: generator.integers(0, 3),
                    2: generator.integers(0, 3),
                    3: generator.integers(0, 3),
                },
                optimizer=stk.MCHammer(),
            ),
        )


def get_key(record: MoleculeRecord) -> str:
    pass


def get_pore_diameter(
    db: atomlite.Database,
    record: MoleculeRecord,
) -> float:
    key = get_key(record)
    path = "$.gokay_ea.pore_diameter"
    pore_diameter = cast(float | None, db.get_property(key, path))
    if pore_diameter is None:
        pw_mol = pywindow.Molecule.load_rdkit_mol(
            record.get_molecule().to_rdkit_mol()
        )
        pore_diameter = pw_mol.calculate_pore_diameter()
        db.set_property(key, path, pore_diameter)
    return pore_diameter


def get_caffeine_binding_energy(
    xtb_path: pathlib.Path,
    xtb_output_dir: pathlib.Path,
    db: atomlite.Database,
    record: MoleculeRecord,
) -> float:
    key = get_key(record)
    path = "$.gokay_ea.caffeine_binding_energy"
    energy = cast(float | None, db.get_property(key, path))
    if energy is None:
        xtb = stko.XTBEnergy(
            xtb_path=str(xtb_path),
            output_dir=xtb_output_dir,
            cycles=1,
        )
        xtb_results = xtb.get_results(record.get_molecule())
        energy = xtb_results.get_total_energy()[0]
        db.set_property(key, path, energy)
    return energy


def get_functional_group_type(building_block: stk.BuildingBlock) -> type:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__


def is_aldehyde(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Aldehyde


def is_amine(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.PrimaryAmino


def get_num_functional_groups(building_block: stk.BuildingBlock) -> int:
    return building_block.get_num_functional_groups()


def get_entry(generation: int, record: MoleculeRecord) -> atomlite.Entry:
    return atomlite.Entry.from_rdkit(
        key=get_key(record),
        molecule=record.get_molecule().to_rdkit_mol(),
        properties={
            "gokay_ea": {
                f"generation_{generation}": None,
                "building_blocks": [
                    stk.Smiles().get_key(bb)
                    for bb in record.get_topology_graph().get_building_blocks()
                ],
            },
        },
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    generator = np.random.default_rng(4)
    db = atomlite.Database(args.database)
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
        get_initial_population(aldehydes[:2], amines[:2], generator)
    )

    fitness_calculator = stk.PropertyVector(
        property_functions=(
            partial(get_pore_diameter, db),
            partial(
                get_caffeine_binding_energy,
                args.xtb,
                args.xtb_output_dir,
                db,
            ),
        ),
    )

    generation_selector: stk.Best[MoleculeRecord] = stk.Best(
        num_batches=4,
        duplicate_molecules=False,
    )
    stk.SelectionPlotter(
        filename="generation_selection",
        selector=generation_selector,
    )

    mutation_selector: stk.Roulette[MoleculeRecord] = stk.Roulette(
        num_batches=1,
        random_seed=generator,
    )
    stk.SelectionPlotter("mutation_selection", mutation_selector)

    crossover_selector: stk.Roulette[MoleculeRecord] = stk.Roulette(
        num_batches=1,
        batch_size=2,
        random_seed=generator,
    )
    stk.SelectionPlotter("crossover_selection", crossover_selector)

    fitness_normalizer: stk.NormalizerSequence[
        MoleculeRecord
    ] = stk.NormalizerSequence(
        fitness_normalizers=(
            stk.DivideByMean(),
            stk.Sum(),
            stk.Power(-1),
        ),
    )
    ea: stk.EvolutionaryAlgorithm[MoleculeRecord] = stk.EvolutionaryAlgorithm(
        num_processes=1,
        initial_population=initial_population,
        fitness_calculator=fitness_calculator,
        mutator=stk.RandomMutator(
            mutators=(
                stk.RandomBuildingBlock(
                    building_blocks=aldehydes,
                    is_replaceable=is_aldehyde,
                    random_seed=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=aldehydes,
                    is_replaceable=is_aldehyde,
                    random_seed=generator,
                ),
                stk.RandomBuildingBlock(
                    building_blocks=amines,
                    is_replaceable=is_amine,
                    random_seed=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=amines,
                    is_replaceable=is_amine,
                    random_seed=generator,
                ),
            ),
            random_seed=generator,
        ),
        crosser=stk.GeneticRecombination(get_num_functional_groups),
        generation_selector=generation_selector,
        mutation_selector=mutation_selector,
        crossover_selector=crossover_selector,
        fitness_normalizer=fitness_normalizer,
    )

    logging.info("Starting EA.")

    generations = []
    fitness_values: dict[MoleculeRecord, Any] = {}
    pore_diamaters = []
    for i, generation in enumerate(ea.get_generations(5)):
        db.update_entries(
            map(partial(get_entry, i), generation.get_molecule_records())
        )
        generations.append(list(generation.get_molecule_records()))
        fitness_values.update(
            (record, fitness_value.raw)
            for record, fitness_value in generation.get_fitness_values().items()
        )
        pore_diamaters.append(
            [
                cast(
                    float,
                    db.get_property(
                        key=get_key(record),
                        path="$.gokay_ea.pore_diamater",
                    ),
                )
                for record in generation.get_molecule_records()
            ]
        )

    # Write the final population.
    writer = stk.MolWriter()
    final_population_directory = pathlib.Path("final_population")
    final_population_directory.mkdir(exist_ok=True, parents=True)
    for i, record in enumerate(generation.get_molecule_records()):
        writer.write(
            molecule=record.get_molecule(),
            path=final_population_directory / f"final_{i}.mol",
        )

    logging.info("Making fitness plot.")

    # Normalize the fitness values across the entire EA before
    # plotting the fitness values.
    normalized_fitness_values = fitness_normalizer.normalize(fitness_values)
    fitness_values_by_generation = [
        [normalized_fitness_values[record] for record in generation]
        for generation in generations
    ]
    fitness_progress = stk.ProgressPlotter(
        property=fitness_values_by_generation,
        y_label="Fitness Value",
    )
    fitness_progress.get_plot_data().to_csv("fitness_progress.csv")
    fitness_progress.write("fitness_progress.png")

    logging.info("Making rotatable bonds plot.")

    pore_diamater_progress = stk.ProgressPlotter(
        property=pore_diamaters,
        y_label="Pore diameter",
    )
    pore_diamater_progress.write("pore_diameter_progress.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "aldehydes",
        help="SMILES of aldehyde building blocks",
        type=pathlib.Path,
    )
    parser.add_argument(
        "amines",
        help="SMILES of amine building blocks",
        type=pathlib.Path,
    )
    parser.add_argument(
        "database",
        help="path to the Atomlite database",
        type=pathlib.Path,
        default=pathlib.Path("gokay_ea.db"),
    )
    parser.add_argument(
        "xtb",
        help="path to the xtb binary",
        type=pathlib.Path,
        default=pathlib.Path("xtb"),
    )
    parser.add_argument(
        "xtb_output_dir",
        help="directory where xtb outputs are stored",
        type=pathlib.Path,
        default=pathlib.Path("xtb_output_dir"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
