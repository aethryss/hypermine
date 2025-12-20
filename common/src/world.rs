use serde::{Deserialize, Serialize};

use crate::light::{BlockLightInfo, LightBehavior, LightValue};

pub type BlockID = u16;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockID,
    pub name: &'static str,
    /// Index into terrain.png atlas (remcpe TEXTURE_* value)
    pub texture_index: u16,
}

pub struct BlockRegistry;

macro_rules! define_blocks {
    ($($variant:ident { id: $id:literal, name: $name:literal, texture: $texture:literal }),+ $(,)?) => {
        #[repr(u16)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum BlockKind {
            $($variant = $id),+
        }

        impl BlockKind {
            #[inline]
            pub const fn id(self) -> BlockID {
                self as BlockID
            }

            #[inline]
            pub const fn name(self) -> &'static str {
                match self {
                    $(BlockKind::$variant => $name),+
                }
            }

            #[inline]
            pub const fn texture_index(self) -> u16 {
                match self {
                    $(BlockKind::$variant => $texture),+
                }
            }

            #[inline]
            pub const fn block(self) -> Block {
                Block {
                    id: self.id(),
                    name: self.name(),
                    texture_index: self.texture_index(),
                }
            }

            #[inline]
            pub fn from_id(id: BlockID) -> Option<Self> {
                match id {
                    $($id => Some(BlockKind::$variant),)+
                    _ => None,
                }
            }

            #[inline]
            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(BlockKind::$variant),)+
                    _ => None,
                }
            }

            #[inline]
            pub fn all() -> &'static [BlockKind] {
                BLOCK_KINDS
            }
        }

        pub const BLOCKS: &[Block] = &[
            $(Block { id: $id, name: $name, texture_index: $texture }),+
        ];

        pub const BLOCK_KINDS: &[BlockKind] = &[
            $(BlockKind::$variant),+
        ];
    };
}

define_blocks! {
    Air { id: 0, name: "Air", texture: 0 }, // TEXTURE_GRASS_TOP (air uses 0)
    Stone { id: 1, name: "Stone", texture: 1 }, // TEXTURE_STONE
    Grass { id: 2, name: "Grass", texture: 0 }, // TEXTURE_GRASS_TOP
    Dirt { id: 3, name: "Dirt", texture: 2 }, // TEXTURE_DIRT
    Cobblestone { id: 4, name: "Cobblestone", texture: 5 }, // TEXTURE_STONE_SLAB_SIDE (closest to cobble)
    WoodPlanks { id: 5, name: "Wood Planks", texture: 4 }, // TEXTURE_PLANKS
    Sapling { id: 6, name: "Sapling", texture: 15 }, // TEXTURE_SAPLING
    Bedrock { id: 7, name: "Bedrock", texture: 17 }, // TEXTURE_BEDROCK
    Water { id: 8, name: "Water", texture: 14 }, // TEXTURE_WATER_STATIC
    WaterStill { id: 9, name: "Water (Still)", texture: 14 },
    Lava { id: 10, name: "Lava", texture: 237 }, // TEXTURE_LAVA
    LavaStill { id: 11, name: "Lava (Still)", texture: 237 },
    Sand { id: 12, name: "Sand", texture: 18 }, // TEXTURE_SAND
    Gravel { id: 13, name: "Gravel", texture: 19 }, // TEXTURE_GRAVEL
    GoldOre { id: 14, name: "Gold Ore", texture: 32 }, // TEXTURE_ORE_GOLD
    IronOre { id: 15, name: "Iron Ore", texture: 33 }, // TEXTURE_ORE_IRON
    CoalOre { id: 16, name: "Coal Ore", texture: 34 }, // TEXTURE_ORE_COAL
    Log { id: 17, name: "Log", texture: 20 }, // TEXTURE_LOG_SIDE
    Leaves { id: 18, name: "Leaves", texture: 53 }, // TEXTURE_LEAVES_OPAQUE
    Sponge { id: 19, name: "Sponge", texture: 49 }, // TEXTURE_SPONGE
    Glass { id: 20, name: "Glass", texture: 50 }, // TEXTURE_GLASS
    LapisOre { id: 21, name: "Lapis Ore", texture: 160 }, // TEXTURE_ORE_LAPIS
    LapisBlock { id: 22, name: "Lapis Block", texture: 144 }, // TEXTURE_LAPIS
    Sandstone { id: 23, name: "Sandstone", texture: 176 }, // TEXTURE_SANDSTONE_TOP
    Cloth { id: 24, name: "Cloth", texture: 64 }, // TEXTURE_CLOTH_64
    Flower { id: 25, name: "Flower", texture: 13 }, // TEXTURE_FLOWER
    Rose { id: 26, name: "Rose", texture: 12 }, // TEXTURE_ROSE
    BrownMushroom { id: 27, name: "Brown Mushroom", texture: 29 }, // TEXTURE_MUSHROOM_BROWN
    RedMushroom { id: 28, name: "Red Mushroom", texture: 28 }, // TEXTURE_MUSHROOM_RED
    GoldBlock { id: 29, name: "Gold Block", texture: 23 }, // TEXTURE_GOLD
    IronBlock { id: 30, name: "Iron Block", texture: 22 }, // TEXTURE_IRON
    StoneSlabDouble { id: 31, name: "Stone Slab (Double)", texture: 6 }, // TEXTURE_STONE_SLAB_TOP
    StoneSlab { id: 32, name: "Stone Slab", texture: 6 },
    Brick { id: 33, name: "Brick", texture: 7 }, // TEXTURE_BRICKS
    Tnt { id: 34, name: "TNT", texture: 8 }, // TEXTURE_TNT_SIDE
    Bookshelf { id: 35, name: "Bookshelf", texture: 35 }, // TEXTURE_BOOKSHELF
    MossStone { id: 36, name: "Moss Stone", texture: 36 }, // TEXTURE_MOSSY_STONE
    Obsidian { id: 37, name: "Obsidian", texture: 37 }, // TEXTURE_OBSIDIAN
    Torch { id: 38, name: "Torch", texture: 80 }, // TEXTURE_TORCH_LIT
    WoodStairs { id: 39, name: "Wood Stairs", texture: 4 }, // TEXTURE_PLANKS
    DiamondOre { id: 40, name: "Diamond Ore", texture: 51 }, // TEXTURE_ORE_EMERALD (closest)
    DiamondBlock { id: 41, name: "Diamond Block", texture: 24 }, // TEXTURE_EMERALD (closest)
    Farmland { id: 42, name: "Farmland", texture: 86 }, // TEXTURE_FARMLAND
    Furnace { id: 43, name: "Furnace", texture: 43 }, // TEXTURE_FURNACE_FRONT
    FurnaceLit { id: 44, name: "Furnace (Lit)", texture: 61 }, // TEXTURE_FURNACE_LIT
    WoodDoor { id: 45, name: "Wood Door", texture: 80 }, // TEXTURE_DOOR_TOP
    Ladder { id: 46, name: "Ladder", texture: 83 }, // TEXTURE_LADDER
    Rail { id: 47, name: "Rail", texture: 112 }, // TEXTURE_RAIL_CURVED
    CobblestoneStairs { id: 48, name: "Cobblestone Stairs", texture: 5 }, // TEXTURE_STONE_SLAB_SIDE
    IronDoor { id: 49, name: "Iron Door", texture: 82 }, // TEXTURE_DOOR_IRON_TOP
    RedstoneOre { id: 50, name: "Redstone Ore", texture: 51 }, // TEXTURE_ORE_RED_STONE
    RedstoneOreLit { id: 51, name: "Redstone Ore (Lit)", texture: 51 },
    Snow { id: 52, name: "Snow", texture: 66 }, // TEXTURE_SNOW
    Ice { id: 53, name: "Ice", texture: 67 }, // TEXTURE_ICE
    SnowBlock { id: 54, name: "Snow Block", texture: 66 },
    Cactus { id: 55, name: "Cactus", texture: 71 }, // TEXTURE_CACTUS_TOP
    Clay { id: 56, name: "Clay", texture: 72 }, // TEXTURE_CLAY
    Reed { id: 57, name: "Reed", texture: 73 }, // TEXTURE_REEDS
    Jukebox { id: 58, name: "Jukebox", texture: 74 }, // TEXTURE_JUKEBOX_SIDE
    Fence { id: 59, name: "Fence", texture: 4 }, // TEXTURE_PLANKS
    Pumpkin { id: 60, name: "Pumpkin", texture: 102 }, // TEXTURE_PUMPKIN_TOP
    PumpkinLit { id: 61, name: "Pumpkin (Lit)", texture: 104 }, // TEXTURE_PUMPKIN_FACE_LIT
    Netherrack { id: 62, name: "Netherrack", texture: 103 }, // TEXTURE_BLOODSTONE
    SoulSand { id: 63, name: "Soul Sand", texture: 104 }, // TEXTURE_SOULSAND
    Glowstone { id: 64, name: "Glowstone", texture: 105 }, // TEXTURE_GLOWSTONE
    Portal { id: 65, name: "Portal", texture: 0 }, // Use air/blank
    JackOLantern { id: 66, name: "Jack O'Lantern", texture: 104 }, // TEXTURE_PUMPKIN_FACE_LIT
    Cake { id: 67, name: "Cake", texture: 119 }, // TEXTURE_CAKE_TOP
    RedstoneRepeater { id: 68, name: "Redstone Repeater", texture: 0 }, // Use air/blank
    InvisibleBedrock { id: 69, name: "Invisible Bedrock", texture: 0 }, // Use air/blank
}

impl BlockKind {
    /// Returns the light properties for this block type.
    ///
    /// This includes both light emission (for light sources like torches, lava, glowstone)
    /// and light behavior (how the block affects light propagation).
    #[inline]
    pub const fn light_info(self) -> BlockLightInfo {
        use BlockKind::*;
        match self {
            // Air is fully transparent to light
            Air => BlockLightInfo::transparent(),

            // Light-emitting blocks
            // Torch: warm yellow-orange light, level 14
            Torch => BlockLightInfo::emitter_transparent(14, 11, 6),

            // Glowstone: bright warm white light, level 15
            Glowstone => BlockLightInfo::emitter(15, 14, 10),

            // Lava: bright orange-red light, level 15
            Lava | LavaStill => BlockLightInfo {
                emission: LightValue::new(15, 8, 2),
                behavior: LightBehavior::Translucent, // Lava glows through itself
            },

            // Lit furnace: dim orange glow, level 13
            FurnaceLit => BlockLightInfo::emitter(13, 8, 3),

            // Jack O'Lantern and lit pumpkin: warm light, level 15
            JackOLantern | PumpkinLit => BlockLightInfo::emitter(15, 12, 6),

            // Lit redstone ore: dim red glow, level 9
            RedstoneOreLit => BlockLightInfo::emitter(9, 1, 1),

            // Portal: purple glow, level 11
            Portal => BlockLightInfo {
                emission: LightValue::new(6, 2, 11),
                behavior: LightBehavior::Translucent,
            },

            // Translucent blocks (light passes with higher decay)
            Water | WaterStill => BlockLightInfo::translucent(),
            Ice => BlockLightInfo::translucent(),
            Glass => BlockLightInfo::translucent(),
            Leaves => BlockLightInfo::translucent(),

            // Cutout/transparent blocks (light passes with normal decay)
            Sapling | Flower | Rose | BrownMushroom | RedMushroom => {
                BlockLightInfo::transparent()
            }
            Ladder | Rail | Fence | Reed | Cactus => BlockLightInfo::transparent(),
            WoodDoor | IronDoor => BlockLightInfo::transparent(), // Doors have gaps
            Snow => BlockLightInfo::transparent(), // Snow layer

            // All other solid blocks are opaque
            _ => BlockLightInfo::opaque(),
        }
    }

    /// Returns true if this block emits light.
    #[inline]
    pub const fn emits_light(self) -> bool {
        self.light_info().emits_light()
    }

    /// Returns the light emission value for this block (zero for non-emitters).
    #[inline]
    pub const fn light_emission(self) -> LightValue {
        self.light_info().emission
    }

    /// Returns the light behavior for this block.
    #[inline]
    pub const fn light_behavior(self) -> LightBehavior {
        self.light_info().behavior
    }
}

impl BlockRegistry {
    #[inline]
    pub fn get_by_id(id: BlockID) -> Option<Block> {
        BlockKind::from_id(id).map(BlockKind::block)
    }

    #[inline]
    pub fn get_by_name(name: &str) -> Option<Block> {
        BlockKind::from_name(name).map(BlockKind::block)
    }

    #[inline]
    pub fn get_texture(id: BlockID) -> u16 {
        BlockKind::from_id(id)
            .map(|kind| kind.texture_index())
            .unwrap_or(0)
    }

    #[inline]
    pub fn get_name(id: BlockID) -> &'static str {
        BlockKind::from_id(id)
            .map(|kind| kind.name())
            .unwrap_or("Unknown")
    }

    #[inline]
    pub fn count() -> usize {
        BLOCKS.len()
    }

    #[inline]
    pub fn all_blocks() -> &'static [Block] {
        BLOCKS
    }

    #[inline]
    pub fn all_kinds() -> &'static [BlockKind] {
        BLOCK_KINDS
    }

    /// Returns the light info for a block ID, defaulting to opaque for unknown blocks.
    #[inline]
    pub fn get_light_info(id: BlockID) -> BlockLightInfo {
        BlockKind::from_id(id)
            .map(|kind| kind.light_info())
            .unwrap_or(BlockLightInfo::opaque())
    }

    /// Returns the light emission for a block ID, or zero for unknown/non-emitting blocks.
    #[inline]
    pub fn get_light_emission(id: BlockID) -> LightValue {
        BlockKind::from_id(id)
            .map(|kind| kind.light_emission())
            .unwrap_or(LightValue::ZERO)
    }

    /// Returns the light behavior for a block ID, defaulting to opaque for unknown blocks.
    #[inline]
    pub fn get_light_behavior(id: BlockID) -> LightBehavior {
        BlockKind::from_id(id)
            .map(|kind| kind.light_behavior())
            .unwrap_or(LightBehavior::Opaque)
    }
}
