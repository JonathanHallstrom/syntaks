use crate::bitboard::Bitboard;
use crate::core::{Direction, Square};

mod naive;

#[cfg(all(feature = "pext", target_feature = "bmi2"))]
mod pext;

pub type Hits = [(u8, Square); Direction::COUNT];

#[must_use]
pub fn find_hits(blockers: Bitboard, start: Square) -> Hits {
    #[cfg(all(feature = "pext", target_feature = "bmi2"))]
    {
        return pext::find_hits_pext(blockers, start);
    }

    naive::find_hits_naive(blockers, start)
}
