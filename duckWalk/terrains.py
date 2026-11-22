"""Terrain sets for training and the held-out terrains used only in evaluation.
Heights are tiny on purpose: the duck is 25 cm tall and lifts its feet about 1-2 cm.
"""

import mjlab.terrains as terrainGen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


# ---- individual terrain types ----

def flat():
    return terrainGen.BoxFlatTerrainCfg()


def bumpyGrid(maxHeight=0.010, platformWidth=1.5):
    # cobblestone-ish boxes with random heights. grid_width must not divide the tile size evenly
    return terrainGen.BoxRandomGridTerrainCfg(
        grid_width=0.45, grid_height_range=(0.0, maxHeight), platform_width=platformWidth
    )


def smoothBumps(maxHeight=0.015, borderWidth=0.0):
    # rolling perlin noise, uneven but no sharp edges. bump size is set in grid cells,
    # so it stays the same on smaller tiles
    return terrainGen.HfPerlinNoiseTerrainCfg(
        height_range=(0.0, maxHeight), scale=8.0, border_width=borderWidth
    )


def slope(maxSlope=0.10, platformWidth=2.0):
    # rise/run, 0.10 is about 5.7 deg. vertical_scale 1 mm keeps it from turning into tiny steps
    return terrainGen.HfPyramidSlopedTerrainCfg(
        slope_range=(0.03, maxSlope), platform_width=platformWidth, vertical_scale=0.001
    )


def waves(maxAmplitude=0.015, numWaves=12, borderWidth=0.0):
    # default vertical_scale is 5 mm, which rounded these waves down to almost flat.
    # numWaves counts waves per tile: 12 over 8 m gives a ~65 cm wavelength, a few strides per wave
    return terrainGen.HfWaveTerrainCfg(
        amplitude_range=(0.0, maxAmplitude),
        num_waves=numWaves,
        vertical_scale=0.001,
        border_width=borderWidth,
    )


def stairs(maxStep=0.015, platformWidth=2.0, borderWidth=1.0):
    # raised pyramid: the duck spawns on the top platform, so this tests walking down steps
    return terrainGen.BoxPyramidStairsTerrainCfg(
        step_height_range=(0.0, maxStep),
        step_width=0.15,
        platform_width=platformWidth,
        border_width=borderWidth,
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
EVAL_TERRAINS = {
    "flat": flat,
    "bumpyGrid": bumpyGrid,
    "smoothBumps": smoothBumps,
    "slope": slope,
    "waves": waves,
    "stairs": lambda: stairs(maxStep=0.020),
    "bumpyGridHard": lambda: bumpyGrid(maxHeight=0.020),
    "smoothBumpsHard": lambda: smoothBumps(maxHeight=0.030),
    "slopeHard": lambda: slope(maxSlope=0.18),
}

# ---- combined courses ----
# a patchwork of small tiles so a duck crosses several terrain types in one episode.
# 8 m tiles are too big for that: at 0.1-0.4 m/s a duck walks 2-8 m in 20 s, starting at a
# tile center. with 2 m tiles it is 1 m from the first seam and crosses one to four of them.
# every tile's outer edge sits at ground level (flat borders or pyramid edges), so the seams
# between neighbors have no ledges. platforms shrink to 0.5 m so they don't fill the tile.
# evaluate these with a longer episode, e.g. --seconds 20
COMBINED_TILE = 2.0


def combinedSeenTiles():
    # the four training types at their training max
    return {
        "flat": flat(),
        "bumpyGrid": bumpyGrid(platformWidth=0.5),
        "smoothBumps": smoothBumps(borderWidth=0.1),
        "slope": slope(platformWidth=0.5),
    }


def combinedAllTiles():
    # training types plus the novel ones, mixed together
    tiles = combinedSeenTiles()
    tiles["waves"] = waves(numWaves=3, borderWidth=0.1)  # 3 waves over 2 m keeps the ~65 cm wavelength
    tiles["stairs"] = stairs(maxStep=0.020, platformWidth=0.5, borderWidth=0.25)
    return tiles


COMBINED_COURSES = {
    "combinedSeen": combinedSeenTiles,
    "combinedAll": combinedAllTiles,
}

# all names evaluate.py and play.py --terrain accept
EVAL_NAMES = list(EVAL_TERRAINS) + list(COMBINED_COURSES)

# every eval terrain falls in one tier:
#   seen     - the same types and heights the duck trains on
#   harder   - training types pushed past the training max (tests extrapolation).
#              at difficulty 0.5 these only match the training max, so evaluate them above 0.5
#   novel    - shapes never seen in training (tests generalization)
#   combined - patchwork courses that mix types, so the duck has to handle transitions
TIERS = {
    "flat": "seen",
    "bumpyGrid": "seen",
    "smoothBumps": "seen",
    "slope": "seen",
    "bumpyGridHard": "harder",
    "smoothBumpsHard": "harder",
    "slopeHard": "harder",
    "waves": "novel",
    "stairs": "novel",
    "combinedSeen": "combined",
    "combinedAll": "combined",
}
# combinedAll contains novel tiles, so it counts as held out. combinedSeen only has
# training types, but on 8 m training tiles a duck rarely crosses into another type,
# so frequent transitions are new to it. count it as held out too
HELD_OUT = {name for name, tier in TIERS.items() if tier != "seen"}
assert set(TIERS) == set(EVAL_NAMES), "every eval terrain needs a tier"


def evalTerrain(name, difficulty):
    if name in COMBINED_COURSES:
        tiles = COMBINED_COURSES[name]()
        for cfg in tiles.values():
            cfg.proportion = 1.0 / len(tiles)  # equal share, placed at random
        return TerrainGeneratorCfg(
            size=(COMBINED_TILE, COMBINED_TILE),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            curriculum=False,
            difficulty_range=(difficulty, difficulty),
            sub_terrains=tiles,
        )
    if name not in EVAL_TERRAINS:
        raise ValueError(f"unknown eval terrain '{name}', pick from {EVAL_NAMES}")
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=4,
        num_cols=4,
        curriculum=False,
        difficulty_range=(difficulty, difficulty),
        sub_terrains={name: EVAL_TERRAINS[name]()},
    )