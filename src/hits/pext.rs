use crate::bitboard::Bitboard;
use crate::core::{Direction, Square};
use crate::hits::naive::find_hits_naive;
use std::arch::x86_64::{_pdep_u64, _pext_u64};

#[derive(Copy, Clone, Debug)]
struct SquareData {
    mask: Bitboard,
    offset: usize,
}

impl SquareData {
    const fn new() -> Self {
        Self {
            mask: Bitboard::empty(),
            offset: 0,
        }
    }
}

struct Data {
    squares: [SquareData; Square::COUNT],
    table_size: usize,
}

const SQUARE_DATA: Data = {
    let mut squares = [SquareData::new(); Square::COUNT];
    let mut table_size = 0;

    let mut idx = 0;
    while let Some(sq) = Square::from_raw(idx) {
        let square_data = &mut squares[sq.idx()];

        let mut dir_idx = 0;
        while let Some(dir) = Direction::from_raw(dir_idx) {
            let mut dir_bb = Bitboard::empty();

            let mut sq = sq;

            while let Some(shifted) = sq.shift_checked(dir) {
                dir_bb.set_sq(shifted);
                sq = shifted;
            }

            square_data.mask = square_data.mask.or(dir_bb.and(Bitboard::edge(dir).cmpl()));

            dir_idx += 1;
        }

        square_data.offset = table_size;
        table_size += 1 << square_data.mask.popcount();

        idx += 1;
    }

    Data {
        squares,
        table_size,
    }
};

#[static_init::dynamic]
static HITS: [super::Hits; SQUARE_DATA.table_size] = {
    let mut result = [[(0, Square::A1); Direction::COUNT]; SQUARE_DATA.table_size];

    for sq in Square::all() {
        let sq_data = &SQUARE_DATA.squares[sq.idx()];
        let mask = sq_data.mask.raw();

        let entries = 1 << sq_data.mask.popcount();
        for i in 0..entries {
            let blockers = Bitboard::from_raw(unsafe { _pdep_u64(i as u64, mask) });
            result[sq_data.offset + i] = find_hits_naive(blockers, sq);
        }
    }

    result
};

#[must_use]
pub(super) fn find_hits_pext(blockers: Bitboard, start: Square) -> super::Hits {
    let sq_data = &SQUARE_DATA.squares[start.idx()];
    let idx = unsafe { _pext_u64(blockers.raw(), sq_data.mask.raw()) } as usize;
    HITS[sq_data.offset + idx]
}
