"""Channel layout for The Well `active_matter` dataset.

The WellDatasetForJEPA yields tensors of shape (C=11, T, H, W) with this fixed
order, determined by iterating `t0_fields` -> `t1_fields` -> `t2_fields` and
flattening each field's component axes:

    idx  channel           group          field       component
    ---  ----------------  -------------  ----------  ----------
    0    phi               t0_fields      concentration scalar
    1    u_1               t1_fields      velocity    (0,)
    2    u_2               t1_fields      velocity    (1,)
    3    D_11              t2_fields      D           (0, 0)
    4    D_12              t2_fields      D           (0, 1)
    5    D_21              t2_fields      D           (1, 0)
    6    D_22              t2_fields      D           (1, 1)
    7    E_11              t2_fields      E           (0, 0)
    8    E_12              t2_fields      E           (0, 1)
    9    E_21              t2_fields      E           (1, 0)
    10   E_22              t2_fields      E           (1, 1)

The tensor-component flatten order is row-major (C-order), matching
`buf.reshape(2F, H, W, dsize)` inside WellDatasetForJEPA.__getitem__.
"""

CHANNEL_NAMES = (
    "phi",
    "u_1", "u_2",
    "D_11", "D_12", "D_21", "D_22",
    "E_11", "E_12", "E_21", "E_22",
)

# Slice ranges per field group. Use with tensor[..., slice, :, :, :] style indexing.
PHI = slice(0, 1)    # concentration
U = slice(1, 3)      # velocity
D = slice(3, 7)      # orientation tensor (2x2 flattened)
E = slice(7, 11)     # strain-rate tensor (2x2 flattened)

# Physical-parameter ordering returned by WellDatasetForJEPA in `physical_params`.
# The dataset iterates `f['scalars'].keys()` skipping 'L', so the surviving order
# on disk is ['alpha', 'zeta'] for active_matter HDF5 files.
PHYSICAL_PARAM_NAMES = ("alpha", "zeta")
ALPHA_IDX = 0
ZETA_IDX = 1

NUM_CHANNELS = 11
