# conda environment (/data/prashasti/evoscope_source/conda_envs/evoscope) 
import os
import sys
import json
import joblib
import shap
import torch
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import defaultdict
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, AutoModel
from scipy.stats import zscore
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        classification_report
        )

# Sets

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "/data/prashasti/evoscope_source/executables/esm2_t6_8M_UR50D"

IUPRED_PATH = "/data/prashasti/evoscope_source/executables/iupred2a/iupred2a.py"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Inputs

print("\nEvoScope Unified Production Engine\n")

file1 = sys.argv[1]
file2 = sys.argv[2]

# Optional labels.csv for training
LABELS_FILE = "labels.csv"

# Read wt

seq_id = []
mut_seq_id = []

with open(file1, "r") as f1:

    res1 = f1.readlines()

    seq_id.append("Wildtype")

    WT_SEQ = "".join(
            line.strip().upper()
            for line in res1
            if not line.startswith(">")
            )

if not set(WT_SEQ).issubset(VALID_AA):

    raise ValueError(
            "WT sequence contains invalid amino acids."
            )

# Read mut

with open(file2, "r") as f2:

    res2 = f2.readlines()

    for i in res2:

        if i.startswith(">"):

            seq_id.append(i.strip()[1:])
            mut_seq_id.append(i.strip()[1:])

    MUT_SEQS = [
            lin.strip().upper()
            for lin in res2
            if not lin.startswith(">")
            ]

# validate

for item in MUT_SEQS:

    if len(WT_SEQ) != len(item):

        raise ValueError(
                "Sequences must be same length."
                )

    if len(WT_SEQ) < 25:

        raise ValueError(
                "Sequence must be at least 25 amino acids."
                )

    if not set(item).issubset(VALID_AA):

        raise ValueError(
                "Mutant sequence contains invalid amino acids."
                )

# Mut position

mut_list = []

for item in MUT_SEQS:

    mutation_positions = [

            i + 1

            for i, (w, m) in enumerate(
                zip(WT_SEQ, item)
                )

            if w != m
            ]

    mut_list.append(mutation_positions)

    if len(mutation_positions) == 0:

        raise ValueError(
                "WT and mutant are identical."
                )

# Load ESM2

print("Loading ESM2 model...")

tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
        )

esm_model = AutoModel.from_pretrained(
        MODEL_DIR,
        output_attentions=True
        )

esm_model.to(DEVICE)
esm_model.eval()

# Embeddings

def get_embeddings(seq):

    inputs = tokenizer(
            seq,
            return_tensors="pt"
            )

    inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
            }

    with torch.no_grad():

        outputs = esm_model(**inputs)

    embeddings = (
            outputs.last_hidden_state
            .squeeze(0)[1:-1]
            .cpu()
            .numpy()
            )

    attentions = (
            outputs.attentions[-1]
            .mean(dim=1)
            .squeeze(0)[1:-1,1:-1]
            .cpu()
            .numpy()
            )

    return embeddings, attentions

# IUPRED2A

def predict_disorder_iupred(sequence):

    with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".fasta",
            delete=False
            ) as tmp:

        tmp.write(">query\n")
        tmp.write(sequence + "\n")

        fasta_path = tmp.name

    cmd = [
            "python3",
            IUPRED_PATH,
            fasta_path,
            "long"
            ]

    result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
            )

    os.remove(fasta_path)

    scores = []

    for line in result.stdout.splitlines():

        if (
                line.startswith("#")
                or len(line.strip()) == 0
                ):
            continue

        parts = line.split()

        if len(parts) >= 3:

            try:
                scores.append(float(parts[2]))

            except:
                pass

    return np.array(scores)

# Global features

def extract_global_features(seq):

    pa = ProteinAnalysis(seq)

    aa_counts = (
            pd.Series(list(seq))
            .value_counts(normalize=True)
            )

    entropy = -np.sum(
            aa_counts * np.log(aa_counts)
            )

    return {

            "Length": len(seq),

            "MolecularWeight":
            pa.molecular_weight(),

            "Aromaticity":
            pa.aromaticity(),

            "InstabilityIndex":
            pa.instability_index(),

            "IsoelectricPoint":
            pa.isoelectric_point(),

            "GRAVY":
            pa.gravy(),

            "Entropy":
            entropy
            }

# Maps

charge_map = {
        "D": -1,
        "E": -1,
        "K": 1,
        "R": 1,
        "H": 0.5
        }

hydro_scale = {

        "A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,
        "C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
        "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,
        "M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
        "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2

        }

# WT analysis

print("Computing WT embeddings...")

wt_embed, wt_attn = get_embeddings(WT_SEQ)

print("Running IUPred2A on WT...")

wt_disorder = predict_disorder_iupred(WT_SEQ)

wt_global = extract_global_features(WT_SEQ)

# Mut analysis

mut_embeds = []
mut_attns = []

delta_norms = []
attention_shifts = []
global_embedding_shifts = []

mut_disorders = []
disorder_shifts = []

charge_shifts = []
hydro_shifts = []

long_range_changes = []

feature_rows = []

for seq_id_mut, mut_seq, mutation_positions in zip(
        mut_seq_id,
        MUT_SEQS,
        mut_list
        ):

    print(f"Processing {seq_id_mut}")

    mut_embed, mut_attn = get_embeddings(mut_seq)

    mut_embeds.append(mut_embed)
    mut_attns.append(mut_attn)

    mut_disorder = predict_disorder_iupred(mut_seq)

    mut_disorders.append(mut_disorder)

    residue_delta = mut_embed - wt_embed

    delta_norm = np.linalg.norm(
            residue_delta,
            axis=1
            )

    delta_norms.append(delta_norm)

    attention_shift = np.abs(
            wt_attn.mean(axis=1)
            - mut_attn.mean(axis=1)
            )

    attention_shifts.append(attention_shift)

    global_embedding_shift = np.linalg.norm(
            wt_embed.mean(axis=0)
            - mut_embed.mean(axis=0)
            )

    global_embedding_shifts.append(
            global_embedding_shift
            )

    disorder_shift = np.mean(
            np.abs(mut_disorder - wt_disorder)
            )

    disorder_shifts.append(disorder_shift)

    wt_charge = np.array([
        charge_map.get(a,0)
        for a in WT_SEQ
        ])

    mut_charge = np.array([
        charge_map.get(a,0)
        for a in mut_seq
        ])

    charge_shift = mut_charge - wt_charge

    charge_shifts.append(charge_shift)

    wt_hydro = np.array([
        hydro_scale[a]
        for a in WT_SEQ
        ])

    mut_hydro = np.array([
        hydro_scale[a]
        for a in mut_seq
        ])

    hydro_shift = mut_hydro - wt_hydro

    hydro_shifts.append(hydro_shift)

    long_range_mask = np.abs(
            np.subtract.outer(
                np.arange(len(WT_SEQ)),
                np.arange(len(WT_SEQ))
                )
            ) > 10

    long_range_change = np.mean(
            np.abs(
                (mut_attn - wt_attn)[long_range_mask]
                )
            )

    long_range_changes.append(
            long_range_change
            )

    # feature matrix

    feature_rows.append({

        "Seq_ID":
        seq_id_mut,

        "NumMutations":
        len(mutation_positions),

        "GlobalEmbeddingShift":
        float(global_embedding_shift),

        "DisorderShift":
        float(disorder_shift),

        "ElectrostaticShift":
        float(np.mean(np.abs(charge_shift))),

        "HydrophobicShift":
        float(np.mean(np.abs(hydro_shift))),

        "LongRangeCoupling":
        float(long_range_change)
        })


feature_df = pd.DataFrame(feature_rows)
feature_mat = feature_df.copy()
feature_mat = round(feature_mat, 4)
feature_mat.columns = ["Seq_ID", "Num_mutations", "Global Structural Displacement",
        "Disorder Shift", "Electrostatic Shift", "Hydrophobic Shift", "Long Range Coupling"]

feature_mat.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_matrix.csv"
            ),
        index=False
        )

# Disorder plot

plt.figure(figsize=(10,4))

plt.plot(
        wt_disorder,
        label="Wildtype",
        linewidth=2
        )

for md, si in zip(
        mut_disorders,
        mut_seq_id
        ):

    plt.plot(
            md,
            label=si,
            alpha=0.7
            )

plt.axhline(
        y=0.5,
        linestyle="--"
        )

plt.xlabel("Residue Position")

plt.ylabel("IUPred Disorder Score")

plt.title(
        "Intrinsic Disorder Profile"
        )

plt.legend()

plt.tight_layout()

plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "intrinsic_disorder_profile.png"
            )
        )

plt.close()

# Hydrophobicity

hydro_matrix = np.vstack(
        hydro_shifts
        )

nonzero_positions = np.where(

        np.abs(hydro_matrix).sum(axis=0) > 0

        )[0]


if len(nonzero_positions) > 0:

    start = max(
            0,
            nonzero_positions.min() - 10
            )

    end = min(
            len(WT_SEQ),
            nonzero_positions.max() + 10
            )

else:

    start = 0
    end = len(WT_SEQ)

cropped_matrix = hydro_matrix[
        :,
        start:end
        ]


seq_window = end - start

fig_width = max(
        10,
        seq_window / 25
        )

fig_height = max(
        2,
        len(MUT_SEQS) * 0.5
        )

plt.figure(
        figsize=(fig_width, fig_height)
        )


im = plt.imshow(

        cropped_matrix,

        aspect="auto",

        cmap="coolwarm",

        interpolation="nearest"
        )


cbar = plt.colorbar(im)

cbar.set_label(
        "Δ Hydrophobicity",
        fontsize=12
        )

plt.yticks(

        range(len(mut_seq_id)),

        mut_seq_id,

        fontsize=10
        )


window_length = end - start

if window_length <= 50:

    step = 5

elif window_length <= 150:

    step = 10

elif window_length <= 300:

    step = 20

elif window_length <= 600:

    step = 50

else:

    step = 100

xticks = np.arange(
        start,
        end,
        step
        )

xtick_labels = xticks + 1

plt.xticks(

        xticks - start,

        xtick_labels,

        rotation=0,

        fontsize=9
        )

plt.xlabel(
        "Residue Position",
        fontsize=12,
        #fontweight="bold"
        )

plt.ylabel(
        "Mutant Sequences",
        fontsize=12,
        #fontweight="bold"
        )

plt.title(
        "Hydrophobicity Redistribution",
        fontsize=14,
        #fontweight="bold"
        )

plt.tight_layout()

plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "hydrophobic_shift_heatmap.png"
            ),

        dpi=600,
        bbox_inches="tight"
        )

plt.close()

# Attention rewiring

attention_diffs = []

for mut_attn in mut_attns:


    attention_diff = np.mean(

        np.abs(
            mut_attn - wt_attn
        ),

        axis=0
    )

    attention_diff = np.log1p(
        attention_diff * 10000
    )

    attention_diffs.append(
        attention_diff
    )


attn_diff_matrix = np.vstack(
    attention_diffs
)

signal_positions = np.where(

    np.abs(attn_diff_matrix).sum(axis=0)

    >

    np.percentile(

        np.abs(attn_diff_matrix).sum(axis=0),

        70
    )

)[0]


if len(signal_positions) > 0:

    start = max(
        0,
        signal_positions.min() - 20
    )

    end = min(
        len(WT_SEQ),
        signal_positions.max() + 20
    )

else:

    start = 0
    end = len(WT_SEQ)

cropped_matrix = attn_diff_matrix[
    :,
    start:end
]

window_length = end - start


fig_width = max(
    10,
    window_length / 25
)

fig_height = max(
    3,
    len(MUT_SEQS) * 0.6
)

plt.figure(
    figsize=(fig_width, fig_height)
)


im = plt.imshow(

    cropped_matrix,

    aspect="auto",

    cmap="turbo",

    interpolation="nearest"
)

cbar = plt.colorbar(im)

cbar.set_label(
    "Log-Scaled Attention Rewiring",
    fontsize=12
)


plt.yticks(

    range(len(mut_seq_id)),

    mut_seq_id,

    fontsize=10
)


if window_length <= 50:

    step = 5

elif window_length <= 150:

    step = 10

elif window_length <= 300:

    step = 20

elif window_length <= 600:

    step = 50

else:

    step = 100

xticks = np.arange(
    start,
    end,
    step
)

xtick_labels = xticks + 1

plt.xticks(

    xticks - start,

    xtick_labels,

    fontsize=9
)

plt.xlabel(
    "Residue Position",
    fontsize=12
)

plt.ylabel(
    "Mutant Sequence",
    fontsize=12
)

plt.title(
    "Attention Rewiring Landscape",
    fontsize=14
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "attention_difference_heatmap.png"
    ),

    dpi=600,
    bbox_inches="tight"
)

plt.close()
# Perturbation propagation

plt.figure(figsize=(8,5))

for delta_norm, si in zip(
        delta_norms,
        mut_seq_id
        ):

    cumulative_shift = np.cumsum(
            delta_norm
            )

    plt.plot(
            cumulative_shift,
            label=si,
            alpha=0.7
            )

plt.xlabel("Residue Position")

plt.ylabel(
        "Cumulative Embedding Perturbation"
        )

plt.title(
        "Perturbation Propagation"
        )

plt.legend()

plt.tight_layout()

plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "perturbation_propagation.png"
            )
        )

plt.close()

#per residue table

for MUT_SEQ, si, delta_norm, attention_shift, mut_disorder, charge_shift, hydro_shift in zip(
        MUT_SEQS,
        mut_seq_id,
        delta_norms,        # list of per-mutant delta_norm arrays
        attention_shifts,   # list of per-mutant attention shift arrays
        mut_disorders,           # list of per-mutant disorder arrays
        charge_shifts,      # list of per-mutant charge shift arrays
        hydro_shifts        # list of per-mutant hydrophobic shift arrays
        ):
    df = pd.DataFrame({
        "Residue_Index": np.arange(1, len(WT_SEQ) + 1),
        "WT": list(WT_SEQ),
        "Mutant": list(MUT_SEQ),
        "EmbeddingDelta": delta_norm,
        "AttentionShift": attention_shift,
        "WT_Disorder": wt_disorder,
        "Mutant_Disorder": mut_disorder,
        "ChargeShift": charge_shift,
        "HydrophobicShift": hydro_shift
        })
    df = round(df, 4)

    df.to_csv(
            os.path.join(OUTPUT_DIR, f"{si}_per_residue_analysis.csv"),
            index=False
            )

# mutation landscape

AA=list("ACDEFGHIKLMNPQRSTVWY")
landscape=[]
for i,wt_aa in enumerate(WT_SEQ):
    for aa in AA:
        if aa==wt_aa: continue
        mutant=WT_SEQ[:i]+aa+WT_SEQ[i+1:]
        gravy_shift=abs(
                ProteinAnalysis(mutant).gravy()
                - wt_global["GRAVY"])
        landscape.append({
            "Position":i+1,
            "WT":wt_aa,
            "Mutant":aa,
            "GRAVY_Shift":gravy_shift
            })

lds = round(pd.DataFrame(landscape), 4)
lds.to_csv(
        os.path.join(OUTPUT_DIR,"mutation_landscape.csv"),
        index=False)

# Saturation mutagenesis

print("Running saturation mutagenesis analysis...")

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

saturation_results = []

TOP_POSITIONS = np.argsort(
        np.mean(wt_attn, axis=0)
        )[-50:]

for pos in TOP_POSITIONS:

    wt_aa = WT_SEQ[pos]

    for aa in AA_LIST:

        if aa == wt_aa:
            continue

        mutant_seq = (

                WT_SEQ[:pos]

                + aa

                + WT_SEQ[pos+1:]

                )

        try:

            mut_embed, mut_attn = get_embeddings(
                    mutant_seq
                    )

            # embedding perturbation

            residue_delta = (
                    mut_embed - wt_embed
                    )

            delta_norm = np.linalg.norm(
                    residue_delta,
                    axis=1
                    )

            mean_delta = float(
                    np.mean(delta_norm)
                    )

            # attention rewiring

            attention_shift = np.mean(
                    np.abs(
                        wt_attn - mut_attn
                        )
                    )

            # disorder shift

            mut_disorder = predict_disorder_iupred(
                    mutant_seq
                    )

            disorder_shift = np.mean(
                    np.abs(
                        mut_disorder - wt_disorder
                        )
                    )

            # composite score

            perturbation_score = (

                    0.5 * mean_delta

                    + 0.3 * attention_shift

                    + 0.2 * disorder_shift

                    )

            saturation_results.append({

                "Position":
                pos + 1,

                "WT":
                wt_aa,

                "Mutant":
                aa,

                "EmbeddingShift":
                mean_delta,

                "AttentionShift":
                attention_shift,

                "DisorderShift":
                disorder_shift,

                "PerturbationScore":
                perturbation_score
                })

        except Exception as e:

            print(
                    f"Skipping mutation "
                    f"{wt_aa}{pos+1}{aa}: {e}"
                    )


saturation_df = round(pd.DataFrame(
    saturation_results
    ), 4)


saturation_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "saturation_mutagenesis.csv"
            ),

        index=False
        )

print(
        "Saturation mutagenesis completed."
        )

# Functional hotspot analysis

print("Computing functional hotspot scores...")

# attention centrality

attention_centrality = np.mean(
        wt_attn,
        axis=0
        )

# long range connectivity
long_range_scores = []

for i in range(len(WT_SEQ)):

    distal_contacts = []

    for j in range(len(WT_SEQ)):

        if abs(i - j) > 10:

            distal_contacts.append(
                    wt_attn[i, j]
                    )

    long_range_scores.append(
            np.mean(distal_contacts)
            )

long_range_scores = np.array(
        long_range_scores
        )

# sensitivity

position_sensitivity = (
        saturation_df
        .groupby("Position")
        ["PerturbationScore"]
        .mean()
        )

position_sensitivity = position_sensitivity.reindex(
        np.arange(1, len(WT_SEQ)+1),
        fill_value=0
        )

position_sensitivity = position_sensitivity.values

attention_z = zscore(
        attention_centrality
        )

long_range_z = zscore(
        long_range_scores
        )

sensitivity_z = zscore(
        position_sensitivity
        )

# hotspot score

hotspot_score = (
        0.4 * attention_z
        + 0.3 * long_range_z
        + 0.3 * sensitivity_z
        )

if len(hotspot_score) > 11:

    hotspot_score = savgol_filter(
            hotspot_score,
            11,
            3
            )

hotspot_df = pd.DataFrame({

    "ResiduePosition": np.arange(1, len(WT_SEQ)+1),

    "Residue": list(WT_SEQ),

    "AttentionCentrality": attention_centrality,

    "LongRangeConnectivity": long_range_scores,

    "MutationalSensitivity": position_sensitivity,

    "HotspotScore": hotspot_score
    })

hotspot_df = round(hotspot_df.sort_values(
    by="HotspotScore",
    ascending=False
    ), 4)

hotspot_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "functional_hotspots.csv"
            ),
        index=False
        )

plt.figure(figsize=(12,5))

plt.plot(
        np.arange(1, len(WT_SEQ)+1),
        hotspot_score,
        linewidth=2
        )

threshold = np.percentile(
        hotspot_score,
        90
        )

hotspots = np.where(
        hotspot_score >= threshold
        )[0]

plt.scatter(
        hotspots + 1,
        hotspot_score[hotspots],
        s=60
        )

for idx in hotspots:

    plt.text(
            idx + 1,
            hotspot_score[idx],
            f"{WT_SEQ[idx]}{idx+1}",
            fontsize=8
            )

plt.xlabel("Residue Position")

plt.ylabel("Functional Hotspot Score")

plt.title(
        "Druggable Functional Hotspots"
        )

plt.tight_layout()

plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "functional_hotspots.png"
            )
        )

plt.close()

# Machine learning inference

MODEL_FILE = "trained_model.pkl"

FEATURE_FILE = "feature_columns.pkl"

print(
        "Loading pretrained ML model..."
        )

model = joblib.load(
        MODEL_FILE
        )

feature_columns = joblib.load(
        FEATURE_FILE
        )


X_pred = feature_df[
        feature_columns
        ]

# predict

pred_probs = model.predict_proba(
        X_pred
        )[:,1]

pred_classes = model.predict(
        X_pred
        )

feature_df[
        "PredictedImpactProbability"
        ] = pred_probs

feature_df[
        "PredictedClass"
        ] = pred_classes

print(
        "Prediction complete."
        )

prediction_df = round(pd.DataFrame({

    "Seq_ID":
    feature_df["Seq_ID"],

    "PredictedProbability":
    pred_probs,

    "PredictedClass":
    pred_classes

    }), 4)

prediction_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "ml_predictions.csv"
            ),

        index=False

        )

# SHAP analysis

print(
        "Running SHAP analysis..."
        )

explainer = shap.TreeExplainer(
        model
        )

shap_values = explainer.shap_values(
        X_pred
        )

if isinstance(shap_values, list):

    shap_plot_values = shap_values[1]

else:

    if len(shap_values.shape) == 3:

        shap_plot_values = shap_values[:, :, 1]

    else:

        shap_plot_values = shap_values

plt.figure(figsize=(10,6))

shap.summary_plot(

        shap_plot_values,

        X_pred,

        show=False

        )

plt.tight_layout()

plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "shap_summary.png"
            )

        )

plt.close()

# Evolution simulation

freqs = []
selection_coeffs = []

for pred_prob in pred_probs:

    selection_coeff = -0.01 * pred_prob

    selection_coeffs.append(
            selection_coeff
            )

    freq = []

    pop = np.zeros(2000)

    for g in range(300):

        mutation_events = (
                np.random.rand(2000) < 1e-4
                )

        pop[
                (pop == 0)
                & mutation_events
                ] = 1

        fitness = np.array([
            1,
            1 + selection_coeff
            ])

        probs = fitness[
                pop.astype(int)
                ]

        probs = probs / probs.sum()

        pop = np.random.choice(
                pop,
                2000,
                p=probs
                )

        freq.append(np.mean(pop))

    freqs.append(freq)

# evolution plot

plt.figure(figsize=(8,5))

for freq, si in zip(
        freqs,
        mut_seq_id
        ):

    plt.plot(
            freq,
            label=si,
            alpha=0.7
            )

plt.xlabel("Generation")

plt.ylabel(
        "Mutant Allele Frequency"
        )

plt.title(
        "Evolutionary Dynamics"
        )

plt.legend()

plt.tight_layout()

plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "evolution_plot.png"
            )
        )

plt.close()
# =========================================================
# GLOBAL SUMMARY + REPORT GENERATION
# =========================================================

glob_sum = pd.DataFrame()

for (

    mutation_positions,
    global_embedding_shift,
    disorder_shift,
    charge_shift,
    hydro_shift,
    long_range_change,
    pred_prob,
    pred_class,
    selection_coeff,
    freq,
    si,
    delta_norm

) in zip(

    mut_list,
    global_embedding_shifts,
    disorder_shifts,
    charge_shifts,
    hydro_shifts,
    long_range_changes,
    pred_probs,
    pred_classes,
    selection_coeffs,
    freqs,
    mut_seq_id,
    delta_norms
):

    #Summary

    summary = {

        "Seq_ID":
            si,

        "Mutation Positions":
            mutation_positions,

        "Number Of Mutations":
            len(mutation_positions),

        "Global Structural Displacement":
            float(global_embedding_shift),

        "Disorder Shift":
            float(disorder_shift),

        "Mean Electrostatic Shift":
            float(np.mean(np.abs(charge_shift))),

        "Mean Hydrophobic Shift":
            float(np.mean(np.abs(hydro_shift))),

        "Long Range Coupling Index":
            float(long_range_change),

        "Predicted Impact Probability":
            float(pred_prob),

        "Predicted Class":
            int(pred_class),

        "Selection Coefficient":
            float(selection_coeff),

        "Final Frequency":
            float(freq[-1]),

        "Mean Embedding Perturbation":
            float(np.mean(delta_norm))
    }

    df = pd.DataFrame([summary])

    glob_sum = pd.concat(

        [glob_sum, df],

        axis=0,

        ignore_index=True
    )

#Reports

    report_lines = []

    report_lines.append(
        "Mutagenix Detailed Biophysical Interpretation Report"
    )

    report_lines.append(
        f"Wild-Type Length: {len(WT_SEQ)}"
    )

    report_lines.append(
        f"Mutation Positions: {mutation_positions}"
    )

    report_lines.append(
        f"Number of Mutations: {len(mutation_positions)}\n"
    )

    report_lines.append(
        "1. GLOBAL STRUCTURAL DISPLACEMENT"
    )

    report_lines.append(
        f"Global embedding shift: {global_embedding_shift:.6f}"
    )

    report_lines.append(
        "Mutation-induced perturbation in contextual protein representation.\n"
    )

    report_lines.append(
        "2. ELECTROSTATIC REDISTRIBUTION"
    )

    report_lines.append(
        f"Mean charge shift: {np.mean(np.abs(charge_shift)):.6f}"
    )

    report_lines.append(
        "Charge redistribution may alter salt bridges and electrostatic stabilization.\n"
    )

    report_lines.append(
        "3. HYDROPHOBIC REDISTRIBUTION"
    )

    report_lines.append(
        f"Mean hydrophobicity shift: {np.mean(np.abs(hydro_shift)):.6f}"
    )

    report_lines.append(
        "Hydrophobic perturbations may affect packing and aggregation.\n"
    )

    report_lines.append(
        "4. LONG-RANGE RESIDUE COUPLING"
    )

    report_lines.append(
        f"Long-range coupling index: {long_range_change:.6f}"
    )

    report_lines.append(
        "Distal communication rewiring detected.\n"
    )

    report_lines.append(
        "5. INTRINSIC DISORDER REDISTRIBUTION"
    )

    report_lines.append(
        f"Mean disorder shift: {disorder_shift:.6f}"
    )

    report_lines.append(
        "Intrinsic disorder redistribution may affect flexibility, phase separation, and binding promiscuity.\n"
    )

    report_lines.append(
        "6. STRUCTURAL PERTURBATION PROPAGATION"
    )

    report_lines.append(
        f"Mean embedding perturbation: {delta_norm.mean():.6f}"
    )

    report_lines.append(
        "Mutation effects propagate along the sequence.\n"
    )

    report_lines.append(
        "7. MACHINE LEARNING IMPACT PREDICTION"
    )

    report_lines.append(
        f"Predicted pathogenicity probability: {pred_prob:.6f}"
    )

    report_lines.append(
        f"Predicted class: {pred_class}\n"
    )

    report_lines.append(
        "8. EVOLUTIONARY CONSEQUENCES"
    )

    report_lines.append(
        f"Selection coefficient: {selection_coeff:.6f}"
    )

    report_lines.append(
        f"Final simulated allele frequency: {freq[-1]:.6f}\n"
    )

    report_path = os.path.join(

        OUTPUT_DIR,

        f"{si}_detailed_biophysical_report.txt"
    )

    with open(report_path, "w") as f:

        for line in report_lines:

            f.write(line + "\n")


# save global summary
glob_sum = round(glob_sum, 4)
glob_sum.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_summary.csv"
            ),
        index=False
        )

print("\nAnalysis Complete.")
print("Results saved in:", OUTPUT_DIR)

