from jarvis.core.atoms import Atoms as JarvisAtoms
from matminer.featurizers.conversions import StrToComposition
import pandas as pd
from ase.neighborlist import neighbor_list
from ase import Atoms
import numpy as np
from typing import List
from dataclasses import dataclass

def convert_atoms_column_to_ase(df, column="atoms"):
    """
    Convert a DataFrame column of JARVIS atom dicts to ASE atoms.

    Parameters:
        df (pd.DataFrame): DataFrame with a column of JARVIS-style atom dicts
        column (str): Column name containing the dict representations

    Returns:
        List of ASE Atoms objects
    """
    ase_atoms_list = []

    for i, row in df.iterrows():
        try:
            jarvis_atoms = JarvisAtoms.from_dict(row[column])
            ase_atoms = jarvis_atoms.ase_converter()
            ase_atoms_list.append(ase_atoms)
        except Exception as e:
            print(f"Failed to convert row {i}: {e}")
            ase_atoms_list.append(None)

    return ase_atoms_list

def add_composition_column(df, atoms_col="Atoms", formula_col="chemical_formula", composition_col="composition"):
    """
    Adds a chemical formula column and a composition column to a DataFrame
    with ASE Atoms objects.

    Parameters:
        df (pd.DataFrame): Input dataframe
        atoms_col (str): Column name containing ASE Atoms objects
        formula_col (str): Name for new column with chemical formulas
        composition_col (str): Name for new column with pymatgen Composition objects

    Returns:
        pd.DataFrame: DataFrame with added columns
    """
    # Step 1: Add chemical_formula column
    df[formula_col] = df[atoms_col].apply(
        lambda atoms: atoms.get_chemical_formula(mode="hill") if atoms is not None else None
    )

    # Step 2: Add composition column using matminer
    df = StrToComposition().featurize_dataframe(df, formula_col, ignore_errors=True)

    return df

def dataframe_to_dict_list(df, atoms_col='Atoms', target_col='n'):
    """
    Convert a DataFrame with ASE Atoms objects and target values into a list of dictionaries.

    Each dictionary has the following keys:
    - 'positions': atomic coordinates (Nx3)
    - 'numbers': atomic numbers (N,)
    - 'cell': unit cell matrix (3x3)
    - 'y': target value from the target column

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ASE Atoms objects and target values.
    atoms_col : str
        Column name for ASE Atoms objects.
    target_col : str
        Column name for target values.

    Returns
    -------
    list of dict
        List of dictionaries representing the atoms and targets.
    """
    dict_list = []

    for _, row in df.iterrows():
        atoms = row[atoms_col]
        entry = {
            'positions': atoms.get_positions().tolist(),      # shape (N, 3)
            'numbers': atoms.get_atomic_numbers().tolist(),   # shape (N,)
            'cell': atoms.get_cell().array.tolist(),          # shape (3, 3)
            'y': row[target_col]                     # scalar
        }
        dict_list.append(entry)

    return dict_list
