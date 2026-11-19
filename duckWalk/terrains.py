"""Terrain sets for training and the held-out terrains used only in evaluation.
Heights are tiny on purpose: the duck is 25 cm tall and lifts its feet about 1-2 cm.
"""

import mjlab.terrains as terrainGen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


# ---- individual terrain types ----

def flat():
    return terrainGen.BoxFlatTerrainCfg()


def bumpyGrid(maxHeight=0.010):
    # cobblestone-ish boxes with random heights. grid_width must not divide 8 m evenly
    return terrainGen.BoxRandomGridTerrainCfg(
        grid_width=0.45, grid_height_range=(0.0, maxHeight), platform_width=1.5
    )


def smoothBumps(maxHeight=0.015):
    # rolling perlin noise, uneven but no sharp edges
    return terrainGen.HfPerlinNoiseTerrainCfg(height_range=(0.0, maxHeight), scale=8.0)


def slope(maxSlope=0.10):
    # rise/run, 0.10 is about 5.7 deg. vertical_scale 1 mm keeps it from turning into tiny steps
    return terrainGen.HfPyramidSlopedTerrainCfg(
        slope_range=(0.03, maxSlope), platform_width=2.0, vertical_scale=0.001
    )


def waves(maxAmplitude=0.015):
    return terrainGen.HfWaveTerrainCfg(amplitude_range=(0.0, maxAmplitude), num_waves=3)


def stairs(maxStep=0.015):
    return terrainGen.BoxPyramidStairsTerrainCfg(
        step_height_range=(0.0, maxStep), step_width=0.15, platform_width=2.0, border_width=1.0
    )


# ---- training sets ----

def trainTerrains(name):
    if name == "flat":
        return None  # envConfig switches to a plain ground plane
    if name == "mixed":
        subTerrains = {
            "flat": flat(),
            "bumpyGrid": bumpyGrid(),
            "smoothBumps": smoothBumps(),
            "slope": slope(),
        }
        proportions = {"flat": 0.2, "bumpyGrid": 0.3, "smoothBumps": 0.3, "slope": 0.2}
        for key, cfg in subTerrains.items():
            cfg.proportion = proportions[key]
        # rows are difficulty levels, the curriculum moves each duck up/down a row
        return TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            curriculum=True,
            sub_terrains=subTerrains,
        )
    raise ValueError(f"unknown terrain set '{name}', pick from ['flat', 'mixed']")


# ---- evaluation ----
# waves and stairs never show up in training. the *Hard ones are training types pushed past
# the max height/slope seen in training.
EVAL_TERRAINS = {
    "flat": flat,
    "bumpyGrid": bumpyGrid,
    "smoothBumps": smoothBumps,
    "slope": slope,
    "waves": waves,
    "stairs": stairs,
    "bumpyGridHard": lambda: bumpyGrid(maxHeight=0.020),
    "smoothBumpsHard": lambda: smoothBumps(maxHeight=0.030),
    "slopeHard": lambda: slope(maxSlope=0.18),
}
HELD_OUT = {"waves", "stairs", "bumpyGridHard", "smoothBumpsHard", "slopeHard"}


def evalTerrain(name, difficulty):
    if name not in EVAL_TERRAINS:
        raise ValueError(f"unknown eval terrain '{name}', pick from {list(EVAL_TERRAINS)}")
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=4,
        num_cols=4,
        curriculum=False,
        difficulty_range=(difficulty, difficulty),
        sub_terrains={name: EVAL_TERRAINS[name]()},
    )
