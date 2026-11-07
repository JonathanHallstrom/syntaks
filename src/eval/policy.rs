use super::Align64;
use super::accumulator::Accumulator;
use super::forward::activate;
use super::nnue::L1_SIZE;
use crate::core::{PieceType, Square, Transform, normalizer};
use crate::position::Position;
use crate::takmove::Move;

pub const MAX_LEGAL: usize = 2048; // seems to work
const SPREAD_PATTERNS: usize = 62;
pub const MOVES: usize = Square::COUNT * (PieceType::COUNT + 2 * SPREAD_PATTERNS);
pub const DIVISOR: f32 = 2048.0;

#[inline]
pub fn move_index(mv: Move, transform: Transform) -> usize {
    let src = mv.sq().transform(transform);
    if !mv.is_spread() {
        return src.idx() * PieceType::COUNT + mv.pt().idx();
    }

    let dst = mv.spread_dest().transform(transform);
    let horizontal = usize::from(src.rank() == dst.rank());
    let mut pattern = mv.pattern() as usize;
    if dst.idx() < src.idx() {
        pattern ^= 63;
    }
    Square::COUNT * PieceType::COUNT + (2 * src.idx() + horizontal) * SPREAD_PATTERNS + pattern - 1
}

#[repr(C, align(64))]
struct PolicyHead {
    weights: [[i8; L1_SIZE]; MOVES],
    biases: [i32; MOVES],
}

static POLICY: PolicyHead = unsafe { std::mem::transmute(*include_bytes!(env!("POLICYFILE"))) };

pub struct Policy {
    activated: Align64<[u8; L1_SIZE]>,
    transform: Transform,
}

impl Policy {
    pub(super) fn new(acc: &Accumulator, pos: &Position) -> Self {
        let mut activated = Align64([0; L1_SIZE]);
        activate(acc, pos, &mut activated);
        Self {
            activated,
            transform: normalizer(pos.cap_sq(pos.stm())),
        }
    }

    #[must_use]
    pub fn logit(&self, mv: Move) -> i32 {
        let output = move_index(mv, self.transform);
        let weights = POLICY.weights[output].iter().zip(&self.activated.0);
        let sum: i32 = weights.map(|(&w, &a)| i32::from(w) * i32::from(a)).sum();
        sum + POLICY.biases[output]
    }
}

pub fn softmax<'a>(moves: &'a [Move], scores: &'a [i32]) -> impl Iterator<Item = (Move, f32)> + 'a {
    assert_eq!(moves.len(), scores.len());
    let mut probabilities = [0.0; MAX_LEGAL];
    let max = scores.iter().max().unwrap();
    let mut sum = 0.0;
    for (&score, probability) in scores.iter().zip(&mut probabilities[..scores.len()]) {
        *probability = ((score - max) as f32 / DIVISOR).exp();
        sum += *probability;
    }
    moves
        .iter()
        .copied()
        .zip(probabilities)
        .map(move |(mv, p)| (mv, p * 1.0 / sum))
}
