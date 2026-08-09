use bullet_lib::game::inputs::SparseInputType;
use bullet_lib::game::outputs::OutputBuckets;
use bullet_lib::value::loader::{GameResult as LoaderResult, LoadableDataType};
use syntaks::board::Position;
use syntaks::core::{Player, Square};
use syntaks::format::GameResult;

pub const NUM_INPUTS: usize = 216;

#[derive(Clone, Copy)]
pub struct TakBoard {
    tops: [u8; 36],
    score: i16,
    result: u8,
    stm: u8,
}

impl TakBoard {
    pub fn new(pos: &Position, p1_eval: i16, p1_result: GameResult) -> Self {
        let stm = pos.stm();

        let mut tops = [0u8; 36];
        for sq in Square::all() {
            if let Some(pt) = pos.stacks().top(sq) {
                let owner = pos.stacks().top_player(sq).unwrap();
                tops[sq.idx()] = 1 + ((pt.raw() << 1) | u8::from(owner != stm));
            }
        }

        let score = (p1_eval as i32 * stm.sign()).clamp(i16::MIN as i32, i16::MAX as i32) as i16;

        let result = match stm {
            Player::P1 => p1_result,
            Player::P2 => p1_result.flip(),
        };

        Self {
            tops,
            score,
            result: result.raw(),
            stm: stm.raw(),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct StmBucket;

impl OutputBuckets<TakBoard> for StmBucket {
    const BUCKETS: usize = 2;

    fn bucket(&self, pos: &TakBoard) -> u8 {
        pos.stm
    }
}

impl LoadableDataType for TakBoard {
    fn score(&self) -> i16 {
        self.score
    }

    fn result(&self) -> LoaderResult {
        [LoaderResult::Loss, LoaderResult::Draw, LoaderResult::Win][self.result as usize]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Tak216;

impl SparseInputType for Tak216 {
    type RequiredDataType = TakBoard;

    fn num_inputs(&self) -> usize {
        NUM_INPUTS
    }

    fn max_active(&self) -> usize {
        36
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &TakBoard, mut f: F) {
        for (sq, &top) in pos.tops.iter().enumerate() {
            if top == 0 {
                continue;
            }

            let top = (top - 1) as usize;
            let ty = top >> 1;
            let color = top & 1;

            f(ty * 72 + color * 36 + sq, ty * 72 + (1 - color) * 36 + sq);
        }
    }

    fn shorthand(&self) -> String {
        "216".to_string()
    }

    fn description(&self) -> String {
        "tops only tak inputs".to_string()
    }
}
