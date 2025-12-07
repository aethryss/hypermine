use serde::{Serialize, Deserialize};
// TileID constants for each tile type for direct use in terrain generation and elsewhere
pub const TILE_ID_AIR: TileID = TileType::Air as TileID;
pub const TILE_ID_STONE: TileID = TileType::Stone as TileID;
pub const TILE_ID_GRASS: TileID = TileType::Grass as TileID;
pub const TILE_ID_DIRT: TileID = TileType::Dirt as TileID;
pub const TILE_ID_COBBLESTONE: TileID = TileType::Cobblestone as TileID;
pub const TILE_ID_WOOD_PLANKS: TileID = TileType::WoodPlanks as TileID;
pub const TILE_ID_SAPLING: TileID = TileType::Sapling as TileID;
pub const TILE_ID_BEDROCK: TileID = TileType::Bedrock as TileID;
pub const TILE_ID_WATER: TileID = TileType::Water as TileID;
pub const TILE_ID_WATER_STILL: TileID = TileType::WaterStill as TileID;
pub const TILE_ID_LAVA: TileID = TileType::Lava as TileID;
pub const TILE_ID_LAVA_STILL: TileID = TileType::LavaStill as TileID;
pub const TILE_ID_SAND: TileID = TileType::Sand as TileID;
pub const TILE_ID_GRAVEL: TileID = TileType::Gravel as TileID;
pub const TILE_ID_GOLD_ORE: TileID = TileType::GoldOre as TileID;
pub const TILE_ID_IRON_ORE: TileID = TileType::IronOre as TileID;
pub const TILE_ID_COAL_ORE: TileID = TileType::CoalOre as TileID;
pub const TILE_ID_LOG: TileID = TileType::Log as TileID;
pub const TILE_ID_LEAVES: TileID = TileType::Leaves as TileID;
pub const TILE_ID_SPONGE: TileID = TileType::Sponge as TileID;
pub const TILE_ID_GLASS: TileID = TileType::Glass as TileID;
pub const TILE_ID_LAPIS_ORE: TileID = TileType::LapisOre as TileID;
pub const TILE_ID_LAPIS_BLOCK: TileID = TileType::LapisBlock as TileID;
pub const TILE_ID_SANDSTONE: TileID = TileType::Sandstone as TileID;
pub const TILE_ID_CLOTH: TileID = TileType::Cloth as TileID;
pub const TILE_ID_FLOWER: TileID = TileType::Flower as TileID;
pub const TILE_ID_ROSE: TileID = TileType::Rose as TileID;
pub const TILE_ID_BROWN_MUSHROOM: TileID = TileType::BrownMushroom as TileID;
pub const TILE_ID_RED_MUSHROOM: TileID = TileType::RedMushroom as TileID;
pub const TILE_ID_GOLD_BLOCK: TileID = TileType::GoldBlock as TileID;
pub const TILE_ID_IRON_BLOCK: TileID = TileType::IronBlock as TileID;
pub const TILE_ID_STONE_SLAB_DOUBLE: TileID = TileType::StoneSlabDouble as TileID;
pub const TILE_ID_STONE_SLAB: TileID = TileType::StoneSlab as TileID;
pub const TILE_ID_BRICK: TileID = TileType::Brick as TileID;
pub const TILE_ID_TNT: TileID = TileType::TNT as TileID;
pub const TILE_ID_BOOKSHELF: TileID = TileType::Bookshelf as TileID;
pub const TILE_ID_MOSS_STONE: TileID = TileType::MossStone as TileID;
pub const TILE_ID_OBSIDIAN: TileID = TileType::Obsidian as TileID;
pub const TILE_ID_TORCH: TileID = TileType::Torch as TileID;
pub const TILE_ID_WOOD_STAIRS: TileID = TileType::WoodStairs as TileID;
pub const TILE_ID_DIAMOND_ORE: TileID = TileType::DiamondOre as TileID;
pub const TILE_ID_DIAMOND_BLOCK: TileID = TileType::DiamondBlock as TileID;
pub const TILE_ID_FARMLAND: TileID = TileType::Farmland as TileID;
pub const TILE_ID_FURNACE: TileID = TileType::Furnace as TileID;
pub const TILE_ID_FURNACE_LIT: TileID = TileType::FurnaceLit as TileID;
pub const TILE_ID_DOOR_WOOD: TileID = TileType::DoorWood as TileID;
pub const TILE_ID_LADDER: TileID = TileType::Ladder as TileID;
pub const TILE_ID_RAIL: TileID = TileType::Rail as TileID;
pub const TILE_ID_COBBLESTONE_STAIRS: TileID = TileType::CobblestoneStairs as TileID;
pub const TILE_ID_DOOR_IRON: TileID = TileType::DoorIron as TileID;
pub const TILE_ID_REDSTONE_ORE: TileID = TileType::RedstoneOre as TileID;
pub const TILE_ID_REDSTONE_ORE_LIT: TileID = TileType::RedstoneOreLit as TileID;
pub const TILE_ID_SNOW: TileID = TileType::Snow as TileID;
pub const TILE_ID_ICE: TileID = TileType::Ice as TileID;
pub const TILE_ID_SNOW_BLOCK: TileID = TileType::SnowBlock as TileID;
pub const TILE_ID_CACTUS: TileID = TileType::Cactus as TileID;
pub const TILE_ID_CLAY: TileID = TileType::Clay as TileID;
pub const TILE_ID_REED: TileID = TileType::Reed as TileID;
pub const TILE_ID_JUKEBOX: TileID = TileType::Jukebox as TileID;
pub const TILE_ID_FENCE: TileID = TileType::Fence as TileID;
pub const TILE_ID_PUMPKIN: TileID = TileType::Pumpkin as TileID;
pub const TILE_ID_PUMPKIN_LIT: TileID = TileType::PumpkinLit as TileID;
pub const TILE_ID_NETHERRACK: TileID = TileType::Netherrack as TileID;
pub const TILE_ID_SOUL_SAND: TileID = TileType::SoulSand as TileID;
pub const TILE_ID_GLOWSTONE: TileID = TileType::Glowstone as TileID;
pub const TILE_ID_PORTAL: TileID = TileType::Portal as TileID;
pub const TILE_ID_JACK_O_LANTERN: TileID = TileType::JackOLantern as TileID;
pub const TILE_ID_CAKE: TileID = TileType::Cake as TileID;
pub const TILE_ID_REDSTONE_REPEATER: TileID = TileType::RedstoneRepeater as TileID;
pub const TILE_ID_INVISIBLE_BEDROCK: TileID = TileType::InvisibleBedrock as TileID;

/// TileID is the integer identifier for a block/tile type, matching remcpe's system.
pub type TileID = u16;

/// List of tile types, matching remcpe's set. These IDs should correspond to the terrain.png tileset.
#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileType {
    Air = 0,
    Stone = 1,
    Grass = 2,
    Dirt = 3,
    Cobblestone = 4,
    WoodPlanks = 5,
    Sapling = 6,
    Bedrock = 7,
    Water = 8,
    WaterStill = 9,
    Lava = 10,
    LavaStill = 11,
    Sand = 12,
    Gravel = 13,
    GoldOre = 14,
    IronOre = 15,
    CoalOre = 16,
    Log = 17,
    Leaves = 18,
    Sponge = 19,
    Glass = 20,
    LapisOre = 21,
    LapisBlock = 22,
    Sandstone = 23,
    Cloth = 24,
    Flower = 25,
    Rose = 26,
    BrownMushroom = 27,
    RedMushroom = 28,
    GoldBlock = 29,
    IronBlock = 30,
    StoneSlabDouble = 31,
    StoneSlab = 32,
    Brick = 33,
    TNT = 34,
    Bookshelf = 35,
    MossStone = 36,
    Obsidian = 37,
    Torch = 38,
    WoodStairs = 39,
    DiamondOre = 40,
    DiamondBlock = 41,
    Farmland = 42,
    Furnace = 43,
    FurnaceLit = 44,
    DoorWood = 45,
    Ladder = 46,
    Rail = 47,
    CobblestoneStairs = 48,
    DoorIron = 49,
    RedstoneOre = 50,
    RedstoneOreLit = 51,
    Snow = 52,
    Ice = 53,
    SnowBlock = 54,
    Cactus = 55,
    Clay = 56,
    Reed = 57,
    Jukebox = 58,
    Fence = 59,
    Pumpkin = 60,
    PumpkinLit = 61,
    Netherrack = 62,
    SoulSand = 63,
    Glowstone = 64,
    Portal = 65,
    JackOLantern = 66,
    Cake = 67,
    RedstoneRepeater = 68,
    InvisibleBedrock = 69,
    // Add more as needed to match remcpe
}

/// Tile struct, representing a block type and its properties (minimal for now)
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    pub id: TileID,
    pub tile_type: TileType,
    pub name: &'static str,
    /// Index into terrain.png atlas (remcpe TEXTURE_* value)
    pub texture_index: u16,
}

// Example registry of tiles (expand as needed)
pub const TILE_REGISTRY: &[Tile] = &[
    Tile { id: 0, tile_type: TileType::Air, name: "Air", texture_index: 0 }, // TEXTURE_GRASS_TOP (air uses 0)
    Tile { id: 1, tile_type: TileType::Stone, name: "Stone", texture_index: 1 }, // TEXTURE_STONE
    Tile { id: 2, tile_type: TileType::Grass, name: "Grass", texture_index: 0 }, // TEXTURE_GRASS_TOP
    Tile { id: 3, tile_type: TileType::Dirt, name: "Dirt", texture_index: 2 }, // TEXTURE_DIRT
    Tile { id: 4, tile_type: TileType::Cobblestone, name: "Cobblestone", texture_index: 5 }, // TEXTURE_STONE_SLAB_SIDE (closest to cobble)
    Tile { id: 5, tile_type: TileType::WoodPlanks, name: "Wood Planks", texture_index: 4 }, // TEXTURE_PLANKS
    Tile { id: 6, tile_type: TileType::Sapling, name: "Sapling", texture_index: 15 }, // TEXTURE_SAPLING
    Tile { id: 7, tile_type: TileType::Bedrock, name: "Bedrock", texture_index: 17 }, // TEXTURE_BEDROCK
    Tile { id: 8, tile_type: TileType::Water, name: "Water", texture_index: 14 }, // TEXTURE_WATER_STATIC
    Tile { id: 9, tile_type: TileType::WaterStill, name: "Water (Still)", texture_index: 14 },
    Tile { id: 10, tile_type: TileType::Lava, name: "Lava", texture_index: 237 }, // TEXTURE_LAVA
    Tile { id: 11, tile_type: TileType::LavaStill, name: "Lava (Still)", texture_index: 237 },
    Tile { id: 12, tile_type: TileType::Sand, name: "Sand", texture_index: 18 }, // TEXTURE_SAND
    Tile { id: 13, tile_type: TileType::Gravel, name: "Gravel", texture_index: 19 }, // TEXTURE_GRAVEL
    Tile { id: 14, tile_type: TileType::GoldOre, name: "Gold Ore", texture_index: 32 }, // TEXTURE_ORE_GOLD
    Tile { id: 15, tile_type: TileType::IronOre, name: "Iron Ore", texture_index: 33 }, // TEXTURE_ORE_IRON
    Tile { id: 16, tile_type: TileType::CoalOre, name: "Coal Ore", texture_index: 34 }, // TEXTURE_ORE_COAL
    Tile { id: 17, tile_type: TileType::Log, name: "Log", texture_index: 20 }, // TEXTURE_LOG_SIDE
    Tile { id: 18, tile_type: TileType::Leaves, name: "Leaves", texture_index: 53 }, // TEXTURE_LEAVES_OPAQUE
    Tile { id: 19, tile_type: TileType::Sponge, name: "Sponge", texture_index: 49 }, // TEXTURE_SPONGE
    Tile { id: 20, tile_type: TileType::Glass, name: "Glass", texture_index: 50 }, // TEXTURE_GLASS
    Tile { id: 21, tile_type: TileType::LapisOre, name: "Lapis Ore", texture_index: 160 }, // TEXTURE_ORE_LAPIS
    Tile { id: 22, tile_type: TileType::LapisBlock, name: "Lapis Block", texture_index: 144 }, // TEXTURE_LAPIS
    Tile { id: 23, tile_type: TileType::Sandstone, name: "Sandstone", texture_index: 176 }, // TEXTURE_SANDSTONE_TOP
    Tile { id: 24, tile_type: TileType::Cloth, name: "Cloth", texture_index: 64 }, // TEXTURE_CLOTH_64
    Tile { id: 25, tile_type: TileType::Flower, name: "Flower", texture_index: 13 }, // TEXTURE_FLOWER
    Tile { id: 26, tile_type: TileType::Rose, name: "Rose", texture_index: 12 }, // TEXTURE_ROSE
    Tile { id: 27, tile_type: TileType::BrownMushroom, name: "Brown Mushroom", texture_index: 29 }, // TEXTURE_MUSHROOM_BROWN
    Tile { id: 28, tile_type: TileType::RedMushroom, name: "Red Mushroom", texture_index: 28 }, // TEXTURE_MUSHROOM_RED
    Tile { id: 29, tile_type: TileType::GoldBlock, name: "Gold Block", texture_index: 23 }, // TEXTURE_GOLD
    Tile { id: 30, tile_type: TileType::IronBlock, name: "Iron Block", texture_index: 22 }, // TEXTURE_IRON
    Tile { id: 31, tile_type: TileType::StoneSlabDouble, name: "Stone Slab (Double)", texture_index: 6 }, // TEXTURE_STONE_SLAB_TOP
    Tile { id: 32, tile_type: TileType::StoneSlab, name: "Stone Slab", texture_index: 6 },
    Tile { id: 33, tile_type: TileType::Brick, name: "Brick", texture_index: 7 }, // TEXTURE_BRICKS
    Tile { id: 34, tile_type: TileType::TNT, name: "TNT", texture_index: 8 }, // TEXTURE_TNT_SIDE
    Tile { id: 35, tile_type: TileType::Bookshelf, name: "Bookshelf", texture_index: 35 }, // TEXTURE_BOOKSHELF
    Tile { id: 36, tile_type: TileType::MossStone, name: "Moss Stone", texture_index: 36 }, // TEXTURE_MOSSY_STONE
    Tile { id: 37, tile_type: TileType::Obsidian, name: "Obsidian", texture_index: 37 }, // TEXTURE_OBSIDIAN
    Tile { id: 38, tile_type: TileType::Torch, name: "Torch", texture_index: 80 }, // TEXTURE_TORCH_LIT
    Tile { id: 39, tile_type: TileType::WoodStairs, name: "Wood Stairs", texture_index: 4 }, // TEXTURE_PLANKS
    Tile { id: 40, tile_type: TileType::DiamondOre, name: "Diamond Ore", texture_index: 51 }, // TEXTURE_ORE_EMERALD (closest)
    Tile { id: 41, tile_type: TileType::DiamondBlock, name: "Diamond Block", texture_index: 24 }, // TEXTURE_EMERALD (closest)
    Tile { id: 42, tile_type: TileType::Farmland, name: "Farmland", texture_index: 86 }, // TEXTURE_FARMLAND
    Tile { id: 43, tile_type: TileType::Furnace, name: "Furnace", texture_index: 43 }, // TEXTURE_FURNACE_FRONT
    Tile { id: 44, tile_type: TileType::FurnaceLit, name: "Furnace (Lit)", texture_index: 61 }, // TEXTURE_FURNACE_LIT
    Tile { id: 45, tile_type: TileType::DoorWood, name: "Wood Door", texture_index: 80 }, // TEXTURE_DOOR_TOP
    Tile { id: 46, tile_type: TileType::Ladder, name: "Ladder", texture_index: 83 }, // TEXTURE_LADDER
    Tile { id: 47, tile_type: TileType::Rail, name: "Rail", texture_index: 112 }, // TEXTURE_RAIL_CURVED
    Tile { id: 48, tile_type: TileType::CobblestoneStairs, name: "Cobblestone Stairs", texture_index: 5 }, // TEXTURE_STONE_SLAB_SIDE
    Tile { id: 49, tile_type: TileType::DoorIron, name: "Iron Door", texture_index: 82 }, // TEXTURE_DOOR_IRON_TOP
    Tile { id: 50, tile_type: TileType::RedstoneOre, name: "Redstone Ore", texture_index: 51 }, // TEXTURE_ORE_RED_STONE
    Tile { id: 51, tile_type: TileType::RedstoneOreLit, name: "Redstone Ore (Lit)", texture_index: 51 },
    Tile { id: 52, tile_type: TileType::Snow, name: "Snow", texture_index: 66 }, // TEXTURE_SNOW
    Tile { id: 53, tile_type: TileType::Ice, name: "Ice", texture_index: 67 }, // TEXTURE_ICE
    Tile { id: 54, tile_type: TileType::SnowBlock, name: "Snow Block", texture_index: 66 },
    Tile { id: 55, tile_type: TileType::Cactus, name: "Cactus", texture_index: 71 }, // TEXTURE_CACTUS_TOP
    Tile { id: 56, tile_type: TileType::Clay, name: "Clay", texture_index: 72 }, // TEXTURE_CLAY
    Tile { id: 57, tile_type: TileType::Reed, name: "Reed", texture_index: 73 }, // TEXTURE_REEDS
    Tile { id: 58, tile_type: TileType::Jukebox, name: "Jukebox", texture_index: 74 }, // TEXTURE_JUKEBOX_SIDE
    Tile { id: 59, tile_type: TileType::Fence, name: "Fence", texture_index: 4 }, // TEXTURE_PLANKS
    Tile { id: 60, tile_type: TileType::Pumpkin, name: "Pumpkin", texture_index: 102 }, // TEXTURE_PUMPKIN_TOP
    Tile { id: 61, tile_type: TileType::PumpkinLit, name: "Pumpkin (Lit)", texture_index: 104 }, // TEXTURE_PUMPKIN_FACE_LIT
    Tile { id: 62, tile_type: TileType::Netherrack, name: "Netherrack", texture_index: 103 }, // TEXTURE_BLOODSTONE
    Tile { id: 63, tile_type: TileType::SoulSand, name: "Soul Sand", texture_index: 104 }, // TEXTURE_SOULSAND
    Tile { id: 64, tile_type: TileType::Glowstone, name: "Glowstone", texture_index: 105 }, // TEXTURE_GLOWSTONE
    Tile { id: 65, tile_type: TileType::Portal, name: "Portal", texture_index: 0 }, // Use air/blank
    Tile { id: 66, tile_type: TileType::JackOLantern, name: "Jack O'Lantern", texture_index: 104 }, // TEXTURE_PUMPKIN_FACE_LIT
    Tile { id: 67, tile_type: TileType::Cake, name: "Cake", texture_index: 119 }, // TEXTURE_CAKE_TOP
    Tile { id: 68, tile_type: TileType::RedstoneRepeater, name: "Redstone Repeater", texture_index: 0 }, // Use air/blank
    Tile { id: 69, tile_type: TileType::InvisibleBedrock, name: "Invisible Bedrock", texture_index: 0 }, // Use air/blank
];
