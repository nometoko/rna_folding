from cif_reader import CifReader

import os


def main():
    data_dir = "../input/stanford-rna-3d-folding/PDB_RNA"
    for filename in os.listdir(data_dir):
        if filename.endswith(".cif"):
            full_path = os.path.join(data_dir, filename)
            print(f"Processing {full_path}")
            reader = CifReader(full_path)
            reader.read()
            reader.write_to_sequence_csv(".")
            print(f"Finished processing {full_path}")


if __name__ == "__main__":
    main()
