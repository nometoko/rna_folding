import os
import pandas as pd
import plotly.graph_objects as go

from biopandas.mmcif import PandasMmcif
from plotly.subplots import make_subplots
from collections import defaultdict


class CifReader:
    """
    Class to read and parse a CIF file.

    method:
        __init__(cif_file: str):
            initialize the CifReader with a CIF file.
            - params:
                cif_file: path to the CIF file.

        read():
            read the CIF file and extract relevant information.

        plot_structure(chain_id: str):
            shows the 3D structure of the RNA sequence.
            - params:
                chain_id: chain id to plot.
    """

    def __init__(self, cif_file):
        self.pmmcif = PandasMmcif().read_mmcif(cif_file)
        if self.pmmcif.df is None:
            raise ValueError("Invalid CIF file or no data found.")

        self.entry_id = self._get_entry_id()

        self.atoms: pd.DataFrame = self.pmmcif.df["ATOM"]
        self.last_revision_date = self._get_lastest_rivision_date()

        # not duplicated and ordered
        self.ordered_chain_set = list(dict.fromkeys(self.atoms["label_asym_id"]))
        self.chain_num = len(self.ordered_chain_set)

        self.features_dict = defaultdict()
        self.plot_color = {
            "A": "red",
            "G": "blue",
            "C": "green",
            "U": "orange",
        }

        self.sequence_csv = "sequence.csv"
        self.label_csv = "label.csv"

    def read(self):
        for chain_idx, chain_id in enumerate(self.ordered_chain_set):
            sequence = self._get_sequence(chain_id, chain_idx)
            temporal_cutoff = self.last_revision_date
            description = self._get_description(chain_idx)

            # -- XYZ COORDINATES ----------
            label = self._get_labels(chain_id, sequence)

            # --  STORE DATA ----------------------------------------
            self.features_dict[(self.entry_id, chain_id)] = {
                "sequence": sequence,
                "temporal_cutoff": temporal_cutoff,
                "description": description,
                "label": label,
            }

        return self.features_dict

    def plot_structure(self, chain_id: str) -> None:
        chain_mask = self.atoms["label_asym_id"] == chain_id
        chain_df = self.atoms[chain_mask]

        target_id = self._get_target_id(chain_id)

        metric_figure = make_subplots(
            rows=1,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "surface"}]
            ],  # First row: 3D surfaces
            subplot_titles=(
                "{} - Original 3D Structure".format(target_id),
                "{} - Average 3D position of each nucleotide".format(target_id),
            ),
        )

        if type(chain_df) is pd.DataFrame:
            xyz_raw = pd.DataFrame(
                {
                    "x": chain_df["Cartn_x"].to_list(),
                    "y": chain_df["Cartn_y"].to_list(),
                    "z": chain_df["Cartn_z"].to_list(),
                    "resname": chain_df["label_comp_id"].to_list(),
                }
            )
            fig1 = self._subplot_structure(xyz_raw, target_id)
            for t in fig1.data:
                metric_figure.append_trace(t, row=1, col=1)

        features = self.features_dict[(self.entry_id, chain_id)]
        xyz_c1 = pd.DataFrame(
            {
                "x": features["label"][:, 0],
                "y": features["label"][:, 1],
                "z": features["label"][:, 2],
                "resname": list(features["sequence"]),
            }
        )

        fig2 = self._subplot_structure(xyz_c1, target_id)

        for t in fig2.data:
            metric_figure.append_trace(t, row=1, col=2)

        metric_figure.show()

    def write_to_sequence_csv(self, file_dir: str):
        for chain_id in self.ordered_chain_set:
            features = self.features_dict[(self.entry_id, chain_id)]
            sequence = features["sequence"]
            target_id = self._get_target_id(chain_id)

            chain_df = pd.DataFrame(
                {
                    "target_id": target_id,
                    "sequence": sequence,
                    "temporal_cutoff": features["temporal_cutoff"],
                    "description": features["description"],
                }
            )

            # Append to CSV file
            file_path = os.path.join(file_dir, self.sequence_csv)
            chain_df.to_csv(
                file_path,
                mode="a",
                header=not pd.io.common.file_exists(file_path),
                index=False,
            )

    def _get_entry_id(self):
        return self.pmmcif.data["entry"]["id"][0]

    def _get_lastest_rivision_date(self):
        revision_date = self.pmmcif.data["pdbx_audit_revision_history"]["revision_date"]
        return revision_date[0]

    def _get_c1_atoms(self) -> pd.DataFrame:
        """
        Get the C1 atoms from the cif file.
        """

        c1_df = self.atoms.copy()

        res_name = c1_df["label_comp_id"]
        chain_id = c1_df["label_asym_id"]
        entity_id = c1_df["label_entity_id"].astype(str)
        # convet seq_id like 45D to 45
        seq_id = c1_df["label_seq_id"].astype(str).replace("D", "", regex=False)

        c1_df["section"] = res_name + chain_id + entity_id + seq_id

        # only keep necessary columns
        c1_mask = c1_df["label_atom_id"] == "C1'"
        c1_df = c1_df[c1_mask]
        c1_df = c1_df[["section", "Cartn_x", "Cartn_y", "Cartn_z"]]

        # if there are multiple C1 atoms for the same residue, take the first one
        if type(c1_df) is not pd.DataFrame:
            raise TypeError("Invalid data type for c1_df. Expected DataFrame.")

        c1_df = c1_df.groupby("section", sort=False, as_index=False).first()

        return c1_df

    def _get_target_id(self, chain_id: str):
        return f"{self.entry_id}_{chain_id}"

    def _get_sequence(self, chain_id: str, chain_idx: int):
        seq_list = self.pmmcif.data["entity_poly"]["pdbx_seq_one_letter_code_can"]

        if self.chain_num > len(seq_list) - 1:
            sequence = seq_list[0].replace("\n", "")
        else:
            sequence = seq_list[chain_idx].replace("\n", "")

        # If sequence is empty in one_letter_code part, extracting sequence from ATOM df
        if sequence == "":
            chain_mask = self.atoms["label_asym_id"] == chain_id
            chain_df = self.atoms[chain_mask]
            sequence_extraction_df = chain_df[["label_comp_id", "label_seq_id"]]

            if type(sequence_extraction_df) is not pd.DataFrame:
                raise TypeError(
                    "Invalid data type for sequence_extraction_df. Expected DataFrame."
                )

            # get list of residue name from chain id and join them
            # label_comp_id: Residue name
            sequence = sequence_extraction_df.groupby(
                "label_seq_id", sort=False, as_index=False
            ).agg(lambda x: list(set(x))[0])["label_comp_id"]
            sequence = "".join(sequence)

        if len(set(sequence) - {"A", "C", "G", "U"}) > 0:
            raise ValueError(
                f"Invalid sequence: {sequence}. Only A, C, G, U are allowed."
            )

        return sequence

    def _get_description(self, chain_idx: int):
        description_temp = []
        for title in list(self.pmmcif.data.keys()):
            if "pdbx_description" in list(self.pmmcif.data[title].keys()):
                pdbx_description = self.pmmcif.data[title]["pdbx_description"]

                if chain_idx > len(pdbx_description) - 1:
                    if pdbx_description[0] is not None:
                        description_temp.append(pdbx_description[0])
                else:
                    if pdbx_description[chain_idx] is not None:
                        description_temp.append(pdbx_description[chain_idx])

        if description_temp != []:
            description = "|".join(description_temp)
        else:
            description = ""

        return description

    def _get_labels(self, chain_id: str, sequence: str):
        c1_df = self._get_c1_atoms()

        label = c1_df.loc[
            c1_df["section"].apply(
                lambda col: self._check_section(col, sequence, chain_id)
            )
        ][["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(dtype="float32")

        assert len(sequence) == len(label)

        return label

    def _subplot_structure(self, sequence_df, sequence_id):
        fig = go.Figure()

        for resname, color in self.plot_color.items():
            subset = sequence_df[sequence_df["resname"] == resname]
            fig.add_trace(
                go.Scatter3d(
                    x=subset["x"],
                    y=subset["y"],
                    z=subset["z"],
                    mode="markers",
                    marker=dict(size=5, color=color),
                    name=resname,
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=sequence_df["x"],
                y=sequence_df["y"],
                z=sequence_df["z"],
                mode="lines",
                line=dict(color="gray", width=2),
                name="RNA Backbone",
            )
        )

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title=f"3D RNA Structure of sequence {sequence_id}",
        )

        return fig

    @staticmethod
    def _check_section(text, sequence, asym_id):
        condition = [text.startswith(letter + asym_id) for letter in sequence]
        return True if sum(condition) else False
