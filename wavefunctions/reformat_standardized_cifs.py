import os

PARENT_DIR = r"Z:\Jeffrey-Ede\crystal_structures\standardized_inorganic_no_H\\"

files = [PARENT_DIR+f for f in os.listdir(PARENT_DIR)]

num_files = len(files)
for i, file in enumerate(files):
    print(f"File {i} of {num_files}")

    with open(file, "r") as f:
        string = f.read()

        source = "_symmetry_cell_setting          triclinic"
        sink = """_symmetry_cell_setting          triclinic
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'"""

        string = string.replace(source, sink)

        source = """  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_U_iso_or_equiv
  _atom_site_adp_type
  _atom_site_occupancy"""
        sink = """_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
_atom_site_adp_type
_atom_site_occupancy"""

        string = string.replace(source, sink)

    with open(file, "w") as f:
        f.write(string)