use bullet_lib::game::inputs::SparseInputType;
use bullet_lib::game::outputs::OutputBuckets;
use bullet_lib::value::loader::{GameResult as LoaderResult, LoadableDataType};
use std::ops::Deref;
use syntaks::core::{Color, Square, normalizer};
use syntaks::eval::nnue::{
    RESERVES_INPUT_SIZE, STACK_INPUT_SIZE, TOP_INPUT_SIZE, reserves_feature_idx, stack_feature_idx, top_feature_idx,
};
use syntaks::format::PackedTakBoard;
use syntaks::position::Position;
use syntaks::takmove::Move;

pub const RESERVE_BUCKETS: usize = 8;
pub const KOMI_BUCKETS: usize = (Position::MAX_HALF_KOMI - Position::MIN_HALF_KOMI) as usize + 1;
pub const OUTPUT_BUCKETS: usize = KOMI_BUCKETS * RESERVE_BUCKETS;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct LoadedTakBoard(pub PackedTakBoard, pub i8, pub Move);

impl Deref for LoadedTakBoard {
    type Target = PackedTakBoard;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl LoadableDataType for LoadedTakBoard {
    fn score(&self) -> i16 {
        self.stm_score() as i16
    }

    fn result(&self) -> LoaderResult {
        [LoaderResult::Loss, LoaderResult::Draw, LoaderResult::Win][self.stm_result().unwrap() as usize]
    }
}

#[derive(Clone, Copy, Default)]
pub struct StmBucket;

impl OutputBuckets<LoadedTakBoard> for StmBucket {
    const BUCKETS: usize = 2;

    fn bucket(&self, pos: &LoadedTakBoard) -> u8 {
        pos.stm().raw()
    }
}

#[derive(Clone, Copy, Default)]
pub struct KomiReserveBuckets;

impl OutputBuckets<LoadedTakBoard> for KomiReserveBuckets {
    const BUCKETS: usize = OUTPUT_BUCKETS;

    fn bucket(&self, pos: &LoadedTakBoard) -> u8 {
        let w_reserves = pos.0.reserves(Color::White);
        let b_reserves = pos.0.reserves(Color::Black);
        let reserves = w_reserves.min(b_reserves);
        let bucket = [
            0, 0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        ][reserves as usize];
        let stm_komi = i32::from(pos.1) * pos.stm().sign();
        let komi_bucket = (-stm_komi - i32::from(Position::MIN_HALF_KOMI)) as usize;
        (komi_bucket * RESERVE_BUCKETS + bucket) as u8
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TakTopRelativeStacks;

impl TakTopRelativeStacks {
    pub const NUM_INPUTS: usize = TOP_INPUT_SIZE + STACK_INPUT_SIZE + RESERVES_INPUT_SIZE;

    pub fn map_position<F: FnMut(usize, usize)>(&self, pos: &Position, mut f: F) {
        let stm = pos.stm();
        let ntm = stm.flip();
        let transforms = [Color::White, Color::Black].map(|perspective| normalizer(pos.cap_sq(perspective)));
        let map_sq = |perspective: Color, sq: Square| sq.transform(transforms[perspective]);

        for sq in pos.occ() {
            let stm_sq = map_sq(stm, sq);
            let ntm_sq = map_sq(ntm, sq);
            let top = pos.top(sq).unwrap();
            f(
                top_feature_idx(stm, top.color(), top.piece_type(), stm_sq),
                top_feature_idx(ntm, top.color(), top.piece_type(), ntm_sq),
            );

            let height = pos.height(sq);
            if height == 1 {
                continue;
            }

            let mut colors = pos.colors(sq) >> 1;
            let count = height.min(13) - 1;
            for depth in 0..count {
                let color = Color::from_raw((colors & 1) as u8).unwrap();
                colors >>= 1;
                f(
                    TOP_INPUT_SIZE + stack_feature_idx(stm, top, color, depth, stm_sq),
                    TOP_INPUT_SIZE + stack_feature_idx(ntm, top, color, depth, ntm_sq),
                );
            }
        }

        for color in [Color::White, Color::Black] {
            let reserves = pos.flats_in_hand(color);
            if reserves == 0 {
                continue;
            }
            assert!(reserves < 30);
            f(
                TOP_INPUT_SIZE + STACK_INPUT_SIZE + reserves_feature_idx(stm, color, reserves),
                TOP_INPUT_SIZE + STACK_INPUT_SIZE + reserves_feature_idx(ntm, color, reserves),
            );
        }
    }
}

impl SparseInputType for TakTopRelativeStacks {
    type RequiredDataType = LoadedTakBoard;

    fn num_inputs(&self) -> usize {
        Self::NUM_INPUTS
    }

    fn max_active(&self) -> usize {
        64
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &LoadedTakBoard, f: F) {
        self.map_position(&pos.decode_position().unwrap(), f);
    }

    fn shorthand(&self) -> String {
        Self::NUM_INPUTS.to_string()
    }

    fn description(&self) -> String {
        "tak top-relative stack inputs".to_string()
    }
}
