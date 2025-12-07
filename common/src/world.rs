use serde::{Serialize, Deserialize};

/// BlockID is the integer identifier for a block type, matching remcpe's system.
pub type BlockID = u16;

/// Block represents a single block type with its metadata
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockID,
    pub name: &'static str,
    /// Index into terrain.png atlas (remcpe TEXTURE_* value)
    pub texture_index: u16,
}

/// Unified block registry containing all block definitions
pub struct BlockRegistry;

impl BlockRegistry {
    /// Get a block by its ID
    pub fn get_by_id(id: BlockID) -> Option<Block> {
        BLOCKS.get(id as usize).copied()
    }

    /// Get a block by its name
    pub fn get_by_name(name: &str) -> Option<Block> {
        BLOCKS.iter().find(|b| b.name == name).copied()
    }

    /// Get texture index for a block ID
    pub fn get_texture(id: BlockID) -> u16 {
        Self::get_by_id(id).map(|b| b.texture_index).unwrap_or(0)
    }

    /// Get block name for a block ID
    pub fn get_name(id: BlockID) -> &'static str {
        Self::get_by_id(id).map(|b| b.name).unwrap_or("Unknown")
    }

    /// Total number of block types
    pub fn count() -> usize {
        BLOCKS.len()
    }

    /// Get all block definitions
    pub fn all_blocks() -> &'static [Block] {
        BLOCKS
    }
}

/// Unified block registry - single source of truth for all block data
pub const BLOCKS: &[Block] = &[
    Block { id: 0, name: "Air", texture_index: 0 }, // TEXTURE_GRASS_TOP (air uses 0)
    Block { id: 1, name: "Stone", texture_index: 1 }, // TEXTURE_STONE
    Block { id: 2, name: "Grass", texture_index: 0 }, // TEXTURE_GRASS_TOP
    Block { id: 3, name: "Dirt", texture_index: 2 }, // TEXTURE_DIRT
    Block { id: 4, name: "Cobblestone", texture_index: 5 }, // TEXTURE_STONE_SLAB_SIDE (closest to cobble)
    Block { id: 5, name: "Wood Planks", texture_index: 4 }, // TEXTURE_PLANKS
    Block { id: 6, name: "Sapling", texture_index: 15 }, // TEXTURE_SAPLING
    Block { id: 7, name: "Bedrock", texture_index: 17 }, // TEXTURE_BEDROCK
    Block { id: 8, name: "Water", texture_index: 14 }, // TEXTURE_WATER_STATIC
    Block { id: 9, name: "Water (Still)", texture_index: 14 },
    Block { id: 10, name: "Lava", texture_index: 237 }, // TEXTURE_LAVA
    Block { id: 11, name: "Lava (Still)", texture_index: 237 },
    Block { id: 12, name: "Sand", texture_index: 18 }, // TEXTURE_SAND
    Block { id: 13, name: "Gravel", texture_index: 19 }, // TEXTURE_GRAVEL
    Block { id: 14, name: "Gold Ore", texture_index: 32 }, // TEXTURE_ORE_GOLD
    Block { id: 15, name: "Iron Ore", texture_index: 33 }, // TEXTURE_ORE_IRON
    Block { id: 16, name: "Coal Ore", texture_index: 34 }, // TEXTURE_ORE_COAL
    Block { id: 17, name: "Log", texture_index: 20 }, // TEXTURE_LOG_SIDE
    Block { id: 18, name: "Leaves", texture_index: 53 }, // TEXTURE_LEAVES_OPAQUE
    Block { id: 19, name: "Sponge", texture_index: 49 }, // TEXTURE_SPONGE
    Block { id: 20, name: "Glass", texture_index: 50 }, // TEXTURE_GLASS
    Block { id: 21, name: "Lapis Ore", texture_index: 160 }, // TEXTURE_ORE_LAPIS
    Block { id: 22, name: "Lapis Block", texture_index: 144 }, // TEXTURE_LAPIS
    Block { id: 23, name: "Sandstone", texture_index: 176 }, // TEXTURE_SANDSTONE_TOP
    Block { id: 24, name: "Cloth", texture_index: 64 }, // TEXTURE_CLOTH_64
    Block { id: 25, name: "Flower", texture_index: 13 }, // TEXTURE_FLOWER
    Block { id: 26, name: "Rose", texture_index: 12 }, // TEXTURE_ROSE
    Block { id: 27, name: "Brown Mushroom", texture_index: 29 }, // TEXTURE_MUSHROOM_BROWN
    Block { id: 28, name: "Red Mushroom", texture_index: 28 }, // TEXTURE_MUSHROOM_RED
    Block { id: 29, name: "Gold Block", texture_index: 23 }, // TEXTURE_GOLD
    Block { id: 30, name: "Iron Block", texture_index: 22 }, // TEXTURE_IRON
    Block { id: 31, name: "Stone Slab (Double)", texture_index: 6 }, // TEXTURE_STONE_SLAB_TOP
    Block { id: 32, name: "Stone Slab", texture_index: 6 },
    Block { id: 33, name: "Brick", texture_index: 7 }, // TEXTURE_BRICKS
    Block { id: 34, name: "TNT", texture_index: 8 }, // TEXTURE_TNT_SIDE
    Block { id: 35, name: "Bookshelf", texture_index: 35 }, // TEXTURE_BOOKSHELF
    Block { id: 36, name: "Moss Stone", texture_index: 36 }, // TEXTURE_MOSSY_STONE
    Block { id: 37, name: "Obsidian", texture_index: 37 }, // TEXTURE_OBSIDIAN
    Block { id: 38, name: "Torch", texture_index: 80 }, // TEXTURE_TORCH_LIT
    Block { id: 39, name: "Wood Stairs", texture_index: 4 }, // TEXTURE_PLANKS
    Block { id: 40, name: "Diamond Ore", texture_index: 51 }, // TEXTURE_ORE_EMERALD (closest)
    Block { id: 41, name: "Diamond Block", texture_index: 24 }, // TEXTURE_EMERALD (closest)
    Block { id: 42, name: "Farmland", texture_index: 86 }, // TEXTURE_FARMLAND
    Block { id: 43, name: "Furnace", texture_index: 43 }, // TEXTURE_FURNACE_FRONT
    Block { id: 44, name: "Furnace (Lit)", texture_index: 61 }, // TEXTURE_FURNACE_LIT
    Block { id: 45, name: "Wood Door", texture_index: 80 }, // TEXTURE_DOOR_TOP
    Block { id: 46, name: "Ladder", texture_index: 83 }, // TEXTURE_LADDER
    Block { id: 47, name: "Rail", texture_index: 112 }, // TEXTURE_RAIL_CURVED
    Block { id: 48, name: "Cobblestone Stairs", texture_index: 5 }, // TEXTURE_STONE_SLAB_SIDE
    Block { id: 49, name: "Iron Door", texture_index: 82 }, // TEXTURE_DOOR_IRON_TOP
    Block { id: 50, name: "Redstone Ore", texture_index: 51 }, // TEXTURE_ORE_RED_STONE
    Block { id: 51, name: "Redstone Ore (Lit)", texture_index: 51 },
    Block { id: 52, name: "Snow", texture_index: 66 }, // TEXTURE_SNOW
    Block { id: 53, name: "Ice", texture_index: 67 }, // TEXTURE_ICE
    Block { id: 54, name: "Snow Block", texture_index: 66 },
    Block { id: 55, name: "Cactus", texture_index: 71 }, // TEXTURE_CACTUS_TOP
    Block { id: 56, name: "Clay", texture_index: 72 }, // TEXTURE_CLAY
    Block { id: 57, name: "Reed", texture_index: 73 }, // TEXTURE_REEDS
    Block { id: 58, name: "Jukebox", texture_index: 74 }, // TEXTURE_JUKEBOX_SIDE
    Block { id: 59, name: "Fence", texture_index: 4 }, // TEXTURE_PLANKS
    Block { id: 60, name: "Pumpkin", texture_index: 102 }, // TEXTURE_PUMPKIN_TOP
    Block { id: 61, name: "Pumpkin (Lit)", texture_index: 104 }, // TEXTURE_PUMPKIN_FACE_LIT
    Block { id: 62, name: "Netherrack", texture_index: 103 }, // TEXTURE_BLOODSTONE
    Block { id: 63, name: "Soul Sand", texture_index: 104 }, // TEXTURE_SOULSAND
    Block { id: 64, name: "Glowstone", texture_index: 105 }, // TEXTURE_GLOWSTONE
    Block { id: 65, name: "Portal", texture_index: 0 }, // Use air/blank
    Block { id: 66, name: "Jack O'Lantern", texture_index: 104 }, // TEXTURE_PUMPKIN_FACE_LIT
    Block { id: 67, name: "Cake", texture_index: 119 }, // TEXTURE_CAKE_TOP
    Block { id: 68, name: "Redstone Repeater", texture_index: 0 }, // Use air/blank
    Block { id: 69, name: "Invisible Bedrock", texture_index: 0 }, // Use air/blank
];
