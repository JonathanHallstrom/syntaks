use crate::bitboard::Bitboard;
use crate::core::{Direction, Square};

mod common;
mod naive;

#[cfg(all(feature = "pext", target_feature = "bmi2"))]
mod pext;

#[cfg(not(all(feature = "pext", target_feature = "bmi2")))]
mod magic;

pub type Hits = [(u8, Square); Direction::COUNT];

#[must_use]
pub fn find_hits(blockers: Bitboard, start: Square) -> Hits {
    #[cfg(all(feature = "pext", target_feature = "bmi2"))]
    {
        return pext::find_hits_pext(blockers, start);
    }

    #[cfg(not(all(feature = "pext", target_feature = "bmi2")))]
    {
        return magic::find_hits_magic(blockers, start);
    }

    unreachable!();
}
